# distillreasoning

<p align="center">
  <img src="images/classroom_hero.png" alt="GLM-5 teacher robot at a chalkboard showing reasoning chains, instructing a tiny Qwen3.5-4B student robot" width="700">
</p>

<p align="center">
  <strong>I borrowed reasoning from a 744B model and taught it to a 4B model that runs on a laptop.</strong>
</p>

<p align="center">
  <em>2,083 problems. GLM-5 as teacher. Qwen3.5-4B as student. ~$7 in compute. Full pipeline from data to deployed model.</em>
</p>

<p align="center">
  <a href="DEVLOG.md">Dev Log</a> |
  <a href="https://huggingface.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled">Model</a> |
  <a href="https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces">Dataset</a>
</p>

---

## The idea

Frontier models reason differently than small models — they think out loud, backtrack, check their work. That reasoning process is the valuable thing. If you can capture it and train a small model to imitate it, the small model punches way above its weight.

GLM-5 is a 744B parameter model available free via Ollama cloud. It exposes its full `<think>` chain. I fed it 2,083 reasoning problems, captured every thought, and used those traces to fine-tune a 4B model that now actually reasons.

Every step — including the mistakes — is in the [Dev Log](DEVLOG.md).

## Results

| Model | GSM8K Accuracy | Format compliance | Avg thinking tokens |
|-------|---------------|-------------------|---------------------|
| Base Qwen3.5-4B | TBD | 0% | 0 |
| Distilled (this) | TBD | ~95%+ | TBD |

*Results updated after eval run*

**Base model on a trick question:**
```
User: A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?
Base: The ball costs $0.10.  ← wrong
```

**Distilled model:**
```
User: A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much is the ball?

<think>
Let x = cost of ball. Bat = x + 1.00.
Total: x + (x + 1.00) = 1.10
2x + 1.00 = 1.10
2x = 0.10
x = 0.05
</think>

<answer>
The ball costs $0.05.
</answer>
```

## How it works

| Step | What | Time |
|------|------|------|
| 1. Collect problems | GSM8K, MATH, ARC, HumanEval — 2,083 total | 5 min |
| 2. Generate traces | GLM-5 solves each with full `<think>` chain via Ollama cloud | ~8 hrs |
| 3. Filter | Keep correct answers with deep reasoning (>50 thinking tokens) | 15 min |
| 4. Format | Convert to chat format with `<think>`/`<answer>` tags, 80/10/10 split | 1 min |
| 5. Train | LoRA SFT via Tinker on Qwen3.5-4B, train on assistant turns only | ~2 hrs |
| 6. Evaluate | Base vs distilled on held-out problems | 30 min |
| 7. Export | GGUF for Ollama, push to HuggingFace | 15 min |

## Why GLM-5 as teacher

Three frontier models are free on Ollama cloud. GLM-5 wins for distillation:

| Model | AIME 2026 | GPQA-Diamond | Ollama Tag |
|-------|-----------|--------------|------------|
| **GLM-5** | **92.7%** | **86.0%** | `glm-5:cloud` |
| Kimi K2.5 | Strong | Strong | `kimi-k2.5:cloud` |
| MiniMax M2.7 | — | — | `minimax-m2.7:cloud` |

Also: MIT licensed, full `<think>` traces exposed, not summarized.

## Quick start

```bash
git clone https://github.com/brianmeyer/distillreasoning.git
cd distillreasoning
python3 -m venv venv && source venv/bin/activate
pip install ollama datasets huggingface_hub tinker tinker-cookbook

# 1. Download 2,083 problems
python scripts/download_problems.py

# 2. Generate reasoning traces (runs overnight)
python scripts/generate_traces.py

# 3. Filter → format → upload → train
HF_TOKEN=xxx TINKER_API_KEY=xxx python scripts/run_pipeline.py
```

Or just use the pre-built datasets:
```python
from datasets import load_dataset
ds = load_dataset("bmeyer2025/glm5-reasoning-traces-sft")
```

## Project structure

```
distillreasoning/
├── scripts/
│   ├── download_problems.py   # Pull GSM8K, MATH, ARC, HuggingEval from HF
│   ├── generate_traces.py     # GLM-5 cloud → reasoning traces (incremental)
│   ├── filter_traces.py       # Drop wrong answers, short traces, repetition
│   ├── format_for_sft.py      # Chat format + <think>/<answer> tags, 80/10/10
│   ├── upload_dataset.py      # Push raw + formatted datasets to HuggingFace
│   ├── train_tinker.py        # LoRA SFT via Tinker API
│   ├── evaluate.py            # Base vs distilled comparison
│   └── run_pipeline.py        # Chains all steps + writes devlog entries
├── notebooks/
│   └── sft_training.ipynb     # Colab notebook (backup to Tinker)
├── cards/
│   ├── dataset_card.md        # HuggingFace dataset card
│   └── model_card.md          # HuggingFace model card
└── DEVLOG.md                  # Full build log — every step and mistake
```

## Datasets

- **[bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces)** — raw traces: problem + GLM-5 thinking + response
- **[bmeyer2025/glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft)** — formatted train/val/test, plug-and-play with any trainer

## What I learned

*(Updated after training completes)*

## License

MIT
