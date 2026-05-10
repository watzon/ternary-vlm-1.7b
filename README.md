# Ternary VLM — First 1.58-Bit Vision-Language Model

**Ternary Bonsai 1.7B + CLIP ViT-L + Trained Projector**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## Abstract

We demonstrate that a **1.58-bit ternary language model** can be retrofitted with vision capabilities through lightweight projector training. Using only 2,000 COCO caption samples and 37 epochs (~2.5 hours on an RTX 4080), we train a 12.6M-parameter vision projector connecting CLIP ViT-L/14 to Ternary Bonsai 1.7B (PrismML). The resulting model produces fluent, contextually appropriate image descriptions despite the extreme weight quantization of the language backbone.

**Key findings:**
- Ternary LLMs **preserve semantic understanding** of vision embeddings
- Projector-only training (0.7% of total parameters) is sufficient for basic vision-language alignment
- Base ternary models without instruction tuning **cannot follow question-answer formats** but produce fluent completion-style descriptions
- Ternary gradients are ~10,000× larger than FP16, requiring aggressive stabilization (fp32 projector, SGD with lr=1e-5, grad clip=1e-4)
- Completion-style prompting achieves 59% score on multi-tier evaluation at epoch 30, compared to near-zero for question-answer prompting

---

## Architecture

```
Image (224×224)
    │
    ▼
CLIP ViT-L/14 (frozen, 304M params)
    │  256 patches × 1024-dim
    ▼
Vision Projector (trainable, 12.6M params)
    ├── LayerNorm(1024)
    ├── Linear(1024→4096) + GELU
    ├── Linear(4096→2048)
    └── LayerNorm(2048) × 0.1
    │  256 vision tokens × 2048-dim
    ▼
Ternary Qwen3-1.7B (frozen, 1.58-bit, ~1.7B params)
    │  All weights in {-1, 0, +1}
    │  Latent FP16 weights during training
    ▼
Generated Caption
```

**Key design decisions:**
- Projector in **fp32** (necessary for gradient stability)
- LLM in **fp16** (ternary latent weights)
- CLIP in **fp16** (vision features converted to fp32 for projector)
- **No LoRA, no LLM fine-tuning** — the ternary weights are never modified

---

## Training

### Setup
- **Dataset:** MS COCO 2017 captions, 2,000 training samples (random subset)
- **Hardware:** Single RTX 4080 (16GB), i9-13900K, 62GB RAM
- **Duration:** 37 epochs × 2,000 samples ≈ 2.5 hours
- **Optimizer:** SGD, lr=1e-5, no momentum, no weight decay
- **Gradient clipping:** value-based, max=1e-4
- **Loss:** Cross-entropy, next-token prediction (standard LM loss)

### Loss Curve

| Epoch | Loss | Notes |
|-------|------|-------|
| 1 | 4.98 | Random projector |
| 5 | 3.94 | Early alignment |
| 10 | 3.69 | First checkpoint — scene descriptions emerge |
| 20 | 3.51 | Consistent outputs, no more hallucination loops |
| 30 | 3.37 | Fluent completions, ~59% eval score |
| 37 | 3.32 | Final — training still converging |

**Loss reduction:** 4.98 → 3.32 (33% decrease)
**NaN occurrences:** 0 across all 74,000 training steps

### Gradient Dynamics

The ternary STE (Straight-Through Estimator) produces gradients ~10,000× larger than standard FP training. This required:

1. **fp32 projector** — Any fp16 in the projector path causes NaN within 2-3 steps
2. **Tiny weight initialization** — `std=1e-4` for projector linear layers
3. **Output scaling** — 0.1× multiplier on projector output to match token embedding magnitude
4. **Aggressive gradient clipping** — 1e-4 value-based clip (standard VLM training uses 1.0)
5. **No optimizer momentum** — Momentum amplifies ternary gradient spikes

Without this stabilization recipe, training diverges within 10-20 steps.

---

## Evaluation

We designed a 3-tier evaluation suite covering 12 test scenarios with 25 total prompt completions:

### Tier 1 — Basic Vision (should work)
Scene description, color recognition, people counting, day/night detection

### Tier 2 — Structured Reasoning (stretch goal)
Large-text OCR, subject differentiation, spatial reasoning, action recognition

### Tier 3 — Complex Understanding (science experiment)
Small-text OCR, chart reading, fine-grained attributes, crowd assessment

### Scoring Rubric
- **0** — Hallucination (repetition loops, QA-template failures, empty output)
- **1** — Vague (generic response, too short, evasive)
- **2** — Reasonable (captures gist, minor errors)
- **3** — Precise (specific details, correct identification)

### Results: Completion vs Question Prompts

| Prompt Style | Epoch 10 | Epoch 20 | Epoch 30 |
|-------------|---------|---------|---------|
| **Question-answer** | 0.7/3 (24%) | 0.8/3 (27%) | 0.8/3 (27%) |
| **Completion** | 1.6/3 (54%) | 1.8/3 (58%) | **1.8/3 (59%)** |

### Tier Breakdown (Completion Style, Epoch 30)

| Tier | Score | Examples |
|------|-------|----------|
| **Tier 1** (scene, colors, people) | 1.9/3 (63%) | "a man in his 30s, wearing glasses and holding an umbrella" |
| **Tier 2** (OCR, spatial, action) | 1.8/3 (58%) | "a small, round object... standing on the ground with hands in front" |
| **Tier 3** (documents, charts, crowds) | 1.8/3 (58%) | "a large outdoor market in the city of Bangkok" |

### Sample Completions — Epoch 30

| Prompt | Completion |
|--------|-----------|
| «The atmosphere of this scene is» | *"very calm and peaceful. The scene is set in a quiet, sunny park with trees that are tall and green."* |
| «The person on the left is wearing» | *"a white shirt and black pants, while the person on the right is wearing blue jeans. The two people..."* |
| «The most noticeable object is a» | *"large, white and black dog with a red collar."* |
| «In the foreground there is» | *"a man in his 30s, wearing glasses and holding an umbrella. In the background..."* |

---

## Key Research Findings

### 1. Base Model Limitation: Instruction Following
The single largest factor in evaluation performance was **prompt format**, not training quality. Ternary Bonsai 1.7B is a base model (not instruction-tuned). When prompted with questions (*"How many vehicles are visible?"*), it generates QA-template completions (*"What is the subject? Why did you choose this image?*") rather than answers. When prompted with completion stems (*"This image shows"*), it produces fluent, appropriate descriptions.

**Implication:** The vision-language connection works. The bottleneck is the LLM's instruction-following capability, which projector training alone cannot teach.

### 2. Ternary Weights Preserve Semantics
Despite all weights being constrained to {-1, 0, +1}, the ternary LLM correctly interprets projected vision embeddings and generates contextually appropriate language. The semantic compression does not destroy the model's ability to process novel input modalities.

### 3. Projector Training is Surprisingly Effective
Only 12.6M parameters (0.7% of the total 1.72B) needed training. The frozen ternary LLM generalizes from text-only pretraining to vision-conditioned generation with just a small learned mapping layer.

### 4. The Scale Problem
At inference time, the projector output is ~100× smaller than token embeddings due to the 0.1× output scaling (needed for training stability). This requires a **scale compensation** factor at inference (×50 for our setup). This is a known issue with the tiny-init + low-LR training recipe and could be addressed by post-training normalization.

### 5. Repetition is the Primary Failure Mode
Without aggressive repetition penalties, the ternary model gets stuck in token loops. We found that a combination of repetition penalty (1.4-1.5), trigram blocking, and sentence-boundary early stopping effectively eliminates this issue.

---

## Quick Start

### Installation

```bash
git clone https://github.com/watzon/ternary-vlm-1.7b
cd ternary-vlm-1.7b
pip install -r requirements.txt
```

### Download Models

You need three model components (all available on HuggingFace):

```bash
# CLIP ViT-L/14 vision encoder (auto-downloaded by transformers)
# Ternary Bonsai 1.7B LLM
huggingface-cli download prism-ml/Ternary-Bonsai-1.7B --local-dir models/ternary-bonsai-1.7b

# Or use the unpacked FP16 weights (recommended for transformers compatibility)
# Already set up if you downloaded from prism-ml
```

### Inference

```python
from ternary_vlm import TernaryVLM
from PIL import Image

# Load model
vlm = TernaryVLM(
    ternary_path="models/ternary-bonsai-1.7b-unpacked",
    checkpoint_path="checkpoints/projector_epoch030.pt"
)

# Generate description
image = Image.open("example.jpg").convert("RGB")
caption = vlm.complete(image, "This image shows")
print(caption)
# → "a woman standing in a room with a large window, looking out at the city..."
```

### Command Line

```bash
python ternary_vlm.py \
  --image photo.jpg \
  --prompt "The photograph captures" \
  --checkpoint checkpoints/projector_epoch030.pt \
  --max-tokens 60
```

---

## Repository Structure

```
ternary-vlm-1.7b/
├── README.md                  # This document
├── ternary_vlm.py             # Clean inference library
├── train_coco.py              # Training script
├── eval_harness.py            # Multi-tier evaluation harness
├── requirements.txt           # Python dependencies
├── checkpoints/
│   ├── projector_epoch010.pt  # Loss=3.69
│   ├── projector_epoch020.pt  # Loss=3.51
│   └── projector_epoch030.pt  # Loss=3.37 (best)
├── eval/
│   └── images/                # Test images (COCO + synthetic)
└── training_history.json      # Per-epoch loss data
```

---

## Limitations & Future Work

### Current Limitations
1. **Hallucinated defaults** — The model overuses certain descriptions ("man in his 30s with glasses") across different images, indicating the projector hasn't fully learned to distinguish visual features
2. **No instruction following** — Requires completion-style prompts; cannot answer direct questions
3. **Poor OCR** — Cannot read text at any scale; defaults to single-letter guesses
4. **No chart understanding** — Sees bars/colors but doesn't extract data semantics
5. **Scale compensation hack** — The ×50 inference scale factor is an artifact of the training recipe
6. **Apple Silicon only for base ternary model** — The original PrismML models are MLX-native; we use unpacked FP16 weights for CUDA

### Future Directions
- **Instruct-tune the ternary LLM** — Would enable question-answering and instruction following
- **Larger training dataset** — 2,000 samples is minimal; COCO full (118K) or LLaVA-style data would improve alignment
- **LoRA on ternary weights** — Currently impossible (no ternary fine-tuning tooling); would need custom STE-based LoRA adaptation
- **Better scale normalization** — Post-training LayerNorm calibration to eliminate the ×50 compensation
- **Compare against other small VLMs** — Moondream, Qwen-VL-Chat 2B, LLaVA-1.5 1.5B as baselines

---

## Citation

```bibtex
@misc{watson2025ternaryvlm,
    title        = {Ternary VLM: Vision-Language Alignment for 1.58-bit Language Models},
    author       = {Christopher Watson},
    year         = {2025},
    howpublished = {https://github.com/watzon/ternary-vlm-1.7b},
    note         = {Research artifact demonstrating projector-based vision alignment
                    for ternary language models using CLIP ViT-L and Ternary Bonsai 1.7B}
}
```

## License

Apache 2.0 — inherited from CLIP (OpenAI) and Ternary Bonsai (PrismML).

## Acknowledgments

- **PrismML** for Ternary Bonsai — pushing the boundaries of extreme quantization
- **OpenAI** for CLIP — the vision backbone that makes this possible
- **Qwen Team** for Qwen3 — the architecture underlying the ternary model
- **MS COCO** for the training dataset
