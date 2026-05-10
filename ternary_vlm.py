#!/usr/bin/env python3
"""
Ternary VLM — Clean Inference Library
======================================
Loads CLIP ViT-L + Vision Projector + Ternary Bonsai 1.7B
and generates image descriptions via completion-style prompting.

Usage:
    from ternary_vlm import TernaryVLM
    vlm = TernaryVLM(ternary_path="models/ternary-bonsai-1.7b-unpacked")
    vlm.load_projector("checkpoints/projector_epoch030.pt")
    caption = vlm.complete(image, "This image shows")

    # CLI: python ternary_vlm.py --image photo.jpg
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import argparse
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LLM_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
PROJ_DTYPE = torch.float32  # MUST be fp32 for stability
INFERENCE_SCALE = 50.0  # Compensate for 0.1× output scale in training
DEFAULT_TERNARY_PATH = "prism-ml/Ternary-Bonsai-1.7B"
DEFAULT_VISION_MODEL = "openai/clip-vit-large-patch14"


# ─── Vision Projector ─────────────────────────────────────────
class VisionProjector(nn.Module):
    """Projects CLIP features (1024-dim) to ternary LLM space (2048-dim).
    
    Architecture: LayerNorm → Linear(1024→4096) → GELU → Linear(4096→2048) → LayerNorm
    Trained in fp32 with output_scale=0.1 for gradient stability.
    At inference, we apply INFERENCE_SCALE to compensate.
    """
    def __init__(self, vision_dim=1024, hidden_dim=4096, llm_dim=2048):
        super().__init__()
        self.norm_in = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, llm_dim)
        self.norm_out = nn.LayerNorm(llm_dim)
    
    def forward(self, x, inference_scale=None):
        x = self.norm_in(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.norm_out(x)
        if inference_scale is not None:
            x = x * inference_scale
        else:
            x = x * 0.1  # Training scale
        return x


# ─── Ternary VLM ──────────────────────────────────────────────
class TernaryVLM:
    """Ternary Vision-Language Model for inference.
    
    Components:
        - CLIP ViT-L/14 (vision encoder, frozen, fp16)
        - Vision Projector (trainable bridge, fp32)
        - Ternary Qwen3-1.7B (language model, frozen, fp16 latent weights)
    
    The LLM uses latent FP16 weights derived from ternary {-1,0,+1} training.
    It functions as a standard Qwen3ForCausalLM for inference purposes.
    """
    
    def __init__(
        self,
        ternary_path: str = DEFAULT_TERNARY_PATH,
        vision_model: str = DEFAULT_VISION_MODEL,
        checkpoint_path: str = None,
    ):
        print(f"Loading Ternary VLM...")
        
        # Vision encoder
        print(f"  Vision: {vision_model}")
        self.vision = CLIPVisionModel.from_pretrained(vision_model).to(DEVICE, LLM_DTYPE).eval()
        self.processor = CLIPImageProcessor.from_pretrained(vision_model)
        for p in self.vision.parameters():
            p.requires_grad = False
        
        # Ternary LLM
        print(f"  LLM: {ternary_path}")
        self.llm = AutoModelForCausalLM.from_pretrained(
            ternary_path, torch_dtype=LLM_DTYPE, local_files_only=("unpacked" in ternary_path)
        ).to(DEVICE).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            ternary_path, local_files_only=("unpacked" in ternary_path)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        for p in self.llm.parameters():
            p.requires_grad = False
        
        vision_dim = self.vision.config.hidden_size  # 1024
        llm_dim = self.llm.config.hidden_size        # 2048
        self.projector = VisionProjector(vision_dim=vision_dim, llm_dim=llm_dim).to(DEVICE, PROJ_DTYPE)
        
        if checkpoint_path:
            self.load_projector(checkpoint_path)
        
        n_vision = sum(p.numel() for p in self.vision.parameters())
        n_llm = sum(p.numel() for p in self.llm.parameters())
        n_proj = sum(p.numel() for p in self.projector.parameters())
        print(f"  Params: {n_vision/1e6:.0f}M (vision) + {n_proj/1e6:.1f}M (projector) + {n_llm/1e9:.2f}B (ternary LLM)")
        print(f"  Ready.")
    
    def load_projector(self, path: str):
        """Load trained projector weights from checkpoint."""
        ckpt = torch.load(path, map_location=DEVICE)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            print(f"  Projector: {path} (epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.4f})")
        else:
            state_dict = ckpt
            print(f"  Projector: {path}")
        self.projector.load_state_dict(state_dict)
        self.projector.eval()
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to vision embeddings in LLM space."""
        if image.size != (224, 224):
            image = image.resize((224, 224))
        
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            features = self.vision(
                inputs["pixel_values"].to(DEVICE, LLM_DTYPE)
            ).last_hidden_state[:, 1:, :].float()  # Remove CLS, fp32 for projector
        
        return self.projector(features, inference_scale=INFERENCE_SCALE).to(LLM_DTYPE)
    
    def complete(
        self,
        image: Image.Image,
        prompt: str = "This image shows",
        max_tokens: int = 80,
        temperature: float = 0.7,
    ) -> str:
        """Generate a completion from a prompt stem.
        
        For best results with this base model, use completion-style prompts:
        - "This image shows" / "The photograph captures"
        - "In the foreground there is" / "Behind the subject"
        - "The atmosphere of this scene is"
        
        Avoid question-style prompts ("How many...?", "What is...?") as the
        base ternary model does not follow instructions.
        """
        # Encode
        vis_embeds = self.encode_image(image)  # (1, 256, 2048)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        full_embeds = torch.cat([vis_embeds, prompt_embeds], dim=1)
        attn_mask = torch.ones(1, full_embeds.shape[1], device=DEVICE, dtype=torch.long)
        
        # Manual autoregressive generation (bypasses HF generate() bug with inputs_embeds)
        past_kv = None
        emb = full_embeds
        am = attn_mask
        generated_ids = []
        
        for _ in range(max_tokens):
            with torch.no_grad():
                out = self.llm(
                    inputs_embeds=emb,
                    attention_mask=am,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            past_kv = out.past_key_values
            
            # Repetition penalty
            logits = out.logits[0, -1, :].clone()
            if generated_ids:
                for tid in set(generated_ids[-15:]):
                    logits[tid] /= 1.4
            
            next_id = logits.argmax().item()
            
            if next_id == self.tokenizer.eos_token_id:
                break
            
            generated_ids.append(next_id)
            
            # Early stop on sentence boundary
            if next_id == self.tokenizer.convert_tokens_to_ids('.'):
                decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                if len(decoded.split()) >= 3:
                    break
            
            # Newline after content = stop
            if next_id == self.tokenizer.convert_tokens_to_ids('\n') and len(generated_ids) > 4:
                break
            
            # Next input
            emb = self.llm.get_input_embeddings()(
                torch.tensor([[next_id]], device=DEVICE)
            )
            am = torch.cat([am, torch.ones(1, 1, device=DEVICE, dtype=torch.long)], dim=1)
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    def describe(self, image: Image.Image, max_tokens: int = 100) -> str:
        """Convenience: describe an image with a standard prompt."""
        return self.complete(image, "This image shows", max_tokens=max_tokens)


# ─── CLI ──────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ternary VLM Inference")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--prompt", default="This image shows", 
                       help="Completion prompt stem")
    parser.add_argument("--checkpoint", default=None,
                       help="Path to projector checkpoint (.pt)")
    parser.add_argument("--ternary-path", default=DEFAULT_TERNARY_PATH,
                       help="Path to ternary LLM")
    parser.add_argument("--max-tokens", type=int, default=80,
                       help="Maximum tokens to generate")
    parser.add_argument("--list-prompts", action="store_true",
                       help="Show example prompts and exit")
    
    args = parser.parse_args()
    
    if args.list_prompts:
        print("Effective completion prompts for ternary VLM:")
        prompts = [
            "This image shows",
            "The photograph captures",
            "In the foreground there is",
            "Behind the subject, there is",
            "The atmosphere of this scene is",
            "The dominant colors in this photo are",
            "The person is holding",
            "This appears to be a",
            "The scene takes place",
        ]
        for p in prompts:
            print(f"  «{p}»")
        exit(0)
    
    vlm = TernaryVLM(
        ternary_path=args.ternary_path,
        checkpoint_path=args.checkpoint,
    )
    
    image = Image.open(args.image).convert("RGB")
    print(f"\nImage: {args.image}")
    print(f"Prompt: «{args.prompt}»")
    print(f"\n--- Completion ---")
    
    result = vlm.complete(image, args.prompt, max_tokens=args.max_tokens)
    print(result)
    print("---")
