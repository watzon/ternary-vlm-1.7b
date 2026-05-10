#!/usr/bin/env python3
"""
Ternary VLM Evaluation Harness v3 — Completion-Style Prompts
=============================================================
Base models don't follow instructions — they complete text.
All prompts rewritten as natural completions the model wants to continue.

Tiers:
  1 — Object presence, scene type, colors, people count, day/night
  2 — OCR (large text), subject differentiation, spatial reasoning, actions
  3 — OCR (small text), document understanding, fine-grained attributes, >5 counting
"""

import torch, torch.nn.functional as F, torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import json, os, sys, argparse, re
from pathlib import Path
from datetime import datetime

DEVICE = "cuda"
DT = torch.float16
SCALE = 50

class VisionProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_in = nn.LayerNorm(1024)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.norm_out = nn.LayerNorm(2048)
    def forward(self, x, sc=0.1):
        return self.norm_out(self.fc2(F.gelu(self.fc1(self.norm_in(x))))) * sc

def load_models():
    vision = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(DEVICE, DT).eval()
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    llm = AutoModelForCausalLM.from_pretrained(
        "/home/watzon/models/ternary-bonsai-1.7b-unpacked", dtype=DT, local_files_only=True
    ).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/watzon/models/ternary-bonsai-1.7b-unpacked", local_files_only=True
    )
    return vision, processor, llm, tokenizer

def load_projector(epoch):
    ckpt = torch.load(
        f"/home/watzon/models/ternary-vlm-output/checkpoints/projector_epoch{epoch:03d}.pt",
        map_location=DEVICE
    )
    proj = VisionProjector().to(DEVICE).eval()
    proj.load_state_dict(ckpt["model_state_dict"])
    return proj, ckpt["loss"]

# ─── Smart Generation ───
def generate(vision, processor, llm, tokenizer, projector, image, prompt, max_tok=60):
    """Completion-style generation — model continues the prompt naturally."""
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        vis_out = vision(inputs["pixel_values"].to(DEVICE, DT))
        features = vis_out.last_hidden_state[:, 1:, :].float()
        vis_embeds = projector(features, sc=SCALE).to(DT)
    
    # Tokenize prompt for completion
    pids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    pembeds = llm.get_input_embeddings()(pids)
    full = torch.cat([vis_embeds, pembeds], dim=1)
    attn = torch.ones(1, full.shape[1], device=DEVICE, dtype=torch.long)
    
    past_kv, emb, am, ids = None, full, attn, []
    for step in range(max_tok):
        with torch.no_grad():
            out = llm(inputs_embeds=emb, attention_mask=am, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        
        logits = out.logits[0, -1, :].clone()
        # Repetition penalty
        if ids:
            for tid in set(ids[-15:]):
                logits[tid] /= 1.4
        
        # Block trigram repeats
        if len(ids) >= 3:
            trigram = tuple(ids[-3:])
            count = 0
            for i in range(len(ids) - 5):
                if tuple(ids[i:i+3]) == trigram:
                    count += 1
                    if count >= 2:
                        if i + 3 < len(ids):
                            logits[ids[i+3]] = -float('inf')
        
        nid = logits.argmax().item()
        if nid == tokenizer.eos_token_id:
            break
        
        ids.append(nid)
        
        # Stop at natural boundaries
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        if (decoded.endswith('.') or decoded.endswith('!') or decoded.endswith('?')) and len(ids) > 8:
            # Give it 3 more tokens then hard stop
            for _ in range(3):
                emb_next = llm.get_input_embeddings()(torch.tensor([[nid]], device=DEVICE))
                am = torch.cat([am, torch.ones(1, 1, device=DEVICE, dtype=torch.long)], dim=1)
                with torch.no_grad():
                    out = llm(inputs_embeds=emb_next, attention_mask=am, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                logits2 = out.logits[0, -1, :].clone()
                if ids:
                    for tid in set(ids[-15:]):
                        logits2[tid] /= 1.4
                nid = logits2.argmax().item()
                if nid == tokenizer.eos_token_id:
                    break
                ids.append(nid)
            break
        
        # Force stop if we hit a newline after content
        if nid == tokenizer.convert_tokens_to_ids('\n') and len(ids) > 6:
            break
        
        emb = llm.get_input_embeddings()(torch.tensor([[nid]], device=DEVICE))
        am = torch.cat([am, torch.ones(1, 1, device=DEVICE, dtype=torch.long)], dim=1)
    
    return tokenizer.decode(ids, skip_special_tokens=True).strip()

# ─── Grading ───
def grade_completion(prompt, completion, expected):
    """Score 0-3 based on completion quality."""
    if not completion or len(completion) < 3:
        return 0, "empty"
    
    # Hallucination patterns
    bad_patterns = [
        r"What is the .{3,30}\?",
        r"Why did you choose",
        r"Please answer in English",
        r"^[A-Z]\. [A-Z]\. [A-Z]\.$",
    ]
    for pat in bad_patterns:
        if re.search(pat, completion):
            return 0, f"hallucination: {pat}"
    
    # Repetition check
    words = completion.split()
    if len(words) > 5:
        for i in range(len(words)-3):
            chunk = " ".join(words[i:i+3])
            if completion.count(chunk) >= 4 and len(chunk) > 8:
                return 0, "repetition"
    
    # Vague answer
    if len(words) < 4:
        return 1, "too short"
    
    vague = ["cannot", "can't tell", "not visible", "i don't know", "unclear"]
    if any(v in completion.lower() for v in vague):
        return 1, "vague"
    
    # Check for concrete content
    has_number = bool(re.search(r'\d+', completion))
    has_color = bool(re.search(r'\b(red|blue|green|white|black|yellow|brown|gray|pink|orange|purple)\b', completion.lower()))
    has_object = bool(re.search(r'\b(car|building|tree|person|people|man|woman|child|dog|cat|table|chair|window|door|street|park|room|sky|water|food|sign|light)\b', completion.lower()))
    has_action = bool(re.search(r'\b(standing|sitting|walking|running|holding|wearing|looking|driving|eating|playing)\b', completion.lower()))
    
    score = 0
    if has_number: score += 1
    if has_color: score += 1
    if has_object: score += 1
    if has_action: score += 1
    
    if score >= 3:
        return 3, "specific details"
    elif score >= 2:
        return 2, "reasonable"
    elif score >= 1:
        return 1, "minimal"
    else:
        return 1, "generic"

# ─── Eval Suite (Completion Prompts!) ───
EVAL_SUITE = {
    "tier1_object": {
        "tier": 1, "category": "Scene Description",
        "image": "tier1/street.jpg",
        "expected": "Street scene with vehicles, buildings, people",
        "prompts": [
            "This image shows",
            "In the foreground there is",
            "The scene takes place",
        ]
    },
    "tier1_colors": {
        "tier": 1, "category": "Color Recognition",
        "image": "tier1/colorful.jpg",
        "expected": "Multiple colors visible",
        "prompts": [
            "The dominant colors in this photo are",
            "The most noticeable object is a",
        ]
    },
    "tier1_people": {
        "tier": 1, "category": "People Description",
        "image": "tier1/people.jpg",
        "expected": "People visible doing something",
        "prompts": [
            "This photograph captures",
            "The people appear to be",
        ]
    },
    "tier1_daynight": {
        "tier": 1, "category": "Time of Day",
        "image": "tier1/night.jpg",
        "expected": "Clear day/night lighting",
        "prompts": [
            "The lighting in this image suggests it is",
            "The atmosphere of this scene is",
        ]
    },
    "tier2_ocr_large": {
        "tier": 2, "category": "OCR (Large Text)",
        "image": "tier2/storefront.jpg",
        "expected": "Business sign or text visible",
        "prompts": [
            "The sign in this image reads",
            "This appears to be a",
        ]
    },
    "tier2_subject": {
        "tier": 2, "category": "Subject Identification",
        "image": "tier2/animals.jpg",
        "expected": "Living things distinguishable",
        "prompts": [
            "The living creatures shown here are",
            "These animals appear to be",
        ]
    },
    "tier2_spatial": {
        "tier": 2, "category": "Spatial Reasoning",
        "image": "tier2/spatial.jpg",
        "expected": "Person holding/carrying something",
        "prompts": [
            "The person is holding",
            "Behind the subject, there is",
        ]
    },
    "tier2_action": {
        "tier": 2, "category": "Action Recognition",
        "image": "tier2/action.jpg",
        "expected": "Clear action being performed",
        "prompts": [
            "The subject is",
            "The posture suggests they are",
        ]
    },
    "tier3_ocr_small": {
        "tier": 3, "category": "Document OCR",
        "image": "tier3/document.jpg",
        "expected": "Document page with text content",
        "prompts": [
            "This document contains",
            "The text on this page reads",
        ]
    },
    "tier3_chart": {
        "tier": 3, "category": "Chart Reading",
        "image": "tier3/chart.jpg",
        "expected": "Bar chart with monthly data",
        "prompts": [
            "This chart displays",
            "The highest value belongs to",
        ]
    },
    "tier3_finegrained": {
        "tier": 3, "category": "Fine-Grained Detail",
        "image": "tier3/group.jpg",
        "expected": "Group with distinctive features",
        "prompts": [
            "The person on the left is wearing",
            "Notable accessories include",
        ]
    },
    "tier3_crowd": {
        "tier": 3, "category": "Crowd Assessment",
        "image": "tier3/crowd.jpg",
        "expected": "Large crowd at an event",
        "prompts": [
            "The crowd in this image appears to be at",
            "The scale of this gathering is",
        ]
    },
}

# ─── Main ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", nargs="+", type=int, default=[10, 20, 30, 50, 100])
    parser.add_argument("--tiers", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", default="/home/watzon/models/ternary-vlm/eval/results")
    parser.add_argument("--image-dir", default="/home/watzon/models/ternary-vlm/eval/images")
    args = parser.parse_args()
    
    print("Loading...")
    vision, processor, llm, tokenizer = load_models()
    all_results = {}
    
    for epoch in args.epochs:
        ckpt_path = f"/home/watzon/models/ternary-vlm-output/checkpoints/projector_epoch{epoch:03d}.pt"
        if not os.path.exists(ckpt_path):
            continue
        
        projector, loss = load_projector(epoch)
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}  (loss={loss:.4f})")
        print(f"{'='*60}")
        
        epoch_results = {"epoch": epoch, "loss": loss, "tests": {}}
        
        for test_id, test in EVAL_SUITE.items():
            if test["tier"] not in args.tiers:
                continue
            
            img_path = os.path.join(args.image_dir, test["image"])
            if not os.path.exists(img_path):
                continue
            
            image = Image.open(img_path).convert("RGB").resize((224, 224))
            
            test_results = {
                "category": test["category"],
                "tier": test["tier"],
                "expected": test["expected"],
                "completions": [],
                "scores": [],
                "total_score": 0,
            }
            
            for prompt in test["prompts"]:
                completion = generate(vision, processor, llm, tokenizer, projector, image, prompt)
                score, rationale = grade_completion(prompt, completion, test["expected"])
                
                test_results["completions"].append({
                    "prompt": prompt,
                    "completion": completion,
                    "score": score,
                    "rationale": rationale
                })
                test_results["scores"].append(score)
                
                marker = ["✗","△","○","●"][score]
                print(f"  [{test_id}] {marker} «{prompt}» {completion[:100]}")
            
            test_results["total_score"] = sum(test_results["scores"])
            test_results["max_score"] = len(test_results["scores"]) * 3
            epoch_results["tests"][test_id] = test_results
        
        all_results[f"epoch_{epoch:03d}"] = epoch_results
    
    # Save
    out_path = os.path.join(args.output, f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print("SCORED SUMMARY (completion-style prompts)")
    print(f"{'='*60}")
    
    for epoch_key, edata in all_results.items():
        ep = edata["epoch"]
        tier_stats = {1: {"scores": [], "total": 0, "max": 0},
                      2: {"scores": [], "total": 0, "max": 0},
                      3: {"scores": [], "total": 0, "max": 0}}
        
        for tid, tdata in edata["tests"].items():
            t = tdata["tier"]
            tier_stats[t]["scores"].extend(tdata["scores"])
            tier_stats[t]["total"] += tdata["total_score"]
            tier_stats[t]["max"] += tdata["max_score"]
        
        print(f"\n  Epoch {ep} (loss={edata['loss']:.4f}):")
        for tier in [1, 2, 3]:
            s = tier_stats[tier]
            if s["scores"]:
                avg = sum(s["scores"]) / len(s["scores"])
                pct = s["total"] / s["max"] * 100
                print(f"    Tier {tier}: {avg:.1f}/3 ({pct:.0f}%) — {s['total']}/{s['max']} across {len(s['scores'])} prompts")
    
    print(f"\n  → {out_path}")

if __name__ == "__main__":
    main()
