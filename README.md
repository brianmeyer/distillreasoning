# distillreasoning

<p align="center">
  <img src="images/classroom_hero.png" alt="GLM-5 teacher robot instructing a tiny Qwen3.5-4B student robot at a chalkboard showing step-by-step reasoning chains" width="700">
</p>

Distill reasoning capabilities from GLM-5 (744B MoE) into a tiny Qwen3.5-4B model that runs anywhere. Zero budget: Ollama cloud for trace generation, Tinker for training.

**The idea:** A frontier model's reasoning isn't just in its answers — it's in how it *thinks*. Capture 2,000+ reasoning traces from GLM-5, train a 4B model to reproduce them, and end up with a small model that reasons far better than its size suggests.

## Deliverables

- **Raw traces:** [bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) — problem + GLM-5 `<think>` chain + response, useful for custom pipelines
- **SFT dataset:** [bmeyer2025/glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft) — formatted train/val/test splits, plug-and-play with any trainer
- **Model:** [bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled](https://huggingface.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled) — distilled model with GGUF exports for local inference
- **Cost:** ~$7 (Ollama cloud free for generation, Tinker for SFT)

## How It Works

<p align="center">
  <img src="images/distillation_brain.png" alt="Large glowing neural network brain streaming reasoning chains into a compact crystal — visualizing knowledge distillation" width="600">
</p>

Knowledge distillation works by having a large "teacher" model generate training data that a small "student" model learns from. The student doesn't learn from raw problems — it learns from the teacher's *reasoning process*.

| Step | What happens |
|------|-------------|
| 1. Collect problems | GSM8K, MATH, ARC-Challenge, HumanEval — 2,083 total |
| 2. Generate traces | GLM-5 solves each with full `<think>` reasoning exposed |
| 3. Filter | Keep only correct answers with deep reasoning (>50 tokens thinking) |
| 4. SFT | Fine-tune Qwen3.5-4B with LoRA — train only on assistant reasoning turns |
| 5. Evaluate | Compare base vs distilled on held-out GSM8K problems |
| 6. Export | GGUF for Ollama/llama.cpp local inference |

## Why GLM-5 as Teacher

Three frontier reasoning models are free via Ollama cloud tags. GLM-5 wins for distillation:

| Model | AIME 2026 | GPQA-Diamond | Ollama Tag |
|-------|-----------|--------------|------------|
| **GLM-5** | **92.7%** | **86.0%** | `glm-5:cloud` |
| Kimi K2.5 | Strong | Strong | `kimi-k2.5:cloud` |
| MiniMax M2.7 | N/A | N/A | `minimax-m2.7:cloud` |

GLM-5 also exposes full `<think>` traces (not summarized) and is MIT licensed — no ambiguity about training on its outputs.

## Quick Start

### 1. Generate Dataset (Local, ~8 hours)

```bash
python3 -m venv venv && source venv/bin/activate
pip install ollama datasets huggingface_hub tinker tinker-cookbook

# Download problems
python scripts/download_problems.py

# Generate traces (runs overnight, saves incrementally)
python scripts/generate_traces.py

# Filter and format into train/val/test splits
python scripts/filter_traces.py
python scripts/format_for_sft.py

# Upload both datasets to HuggingFace
HF_TOKEN=your_token python scripts/upload_dataset.py
```

### 2. Train on Tinker

```bash
TINKER_API_KEY=your_key python scripts/train_tinker.py
```

Or use the Colab notebook as an alternative: `notebooks/sft_training.ipynb`

### 3. Run Locally

```bash
# After GGUF is published to HuggingFace
ollama run hf.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled
```

## Project Structure

```
distillreasoning/
├── scripts/
│   ├── download_problems.py   # Download problem sets from HuggingFace
│   ├── generate_traces.py     # Generate reasoning traces via GLM-5 cloud
│   ├── filter_traces.py       # Filter for quality and correctness
│   ├── format_for_sft.py      # Format into chat format, 80/10/10 split
│   ├── upload_dataset.py      # Push both datasets to HuggingFace
│   ├── train_tinker.py        # SFT training via Tinker API
│   └── evaluate.py            # Compare base vs distilled model
├── notebooks/
│   └── sft_training.ipynb     # Colab notebook (alternative to Tinker)
├── images/
│   ├── classroom_hero.png
│   ├── distillation_brain.png
│   └── distillation_apparatus.png
└── DEVLOG.md                  # Full build log
```

## Dev Log

Full build process including mistakes and decisions: [DEVLOG.md](DEVLOG.md)

## License

MIT
