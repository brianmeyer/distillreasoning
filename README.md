# distillreasoning

<p align="center">
  <img src="images/triptych_hero.png" alt="Three-panel triptych: GLM-5 teacher robot with dense chalkboard, Kimi teacher robot with concise chalkboard, both teachers graduating the tiny Qwen student robot" width="700">
</p>

<p align="center">
  <strong>I borrowed reasoning from a 744B and a 1T model and taught it to a 4B model that runs on a laptop.</strong>
</p>

<p align="center">
  <em>2,083 problems. GLM-5 + Kimi K2.5 as teachers. Qwen3.5-4B as student. ~$14 in compute. Two distilled models, one controlled experiment.</em>
</p>

<p align="center">
  <a href="DEVLOG.md">Dev Log</a> |
  <a href="https://huggingface.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled">Model</a> |
  <a href="https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces">Dataset</a>
</p>

---

## The idea

Frontier models reason differently than small models — they think out loud, backtrack, check their work. That reasoning process is the valuable thing. If you can capture it and train a small model to imitate it, the small model punches way above its weight.

But does it matter *which* frontier model you learn from? GLM-5 writes verbose, detailed reasoning chains. Kimi K2.5 is more concise and elegant. Same problems, same student — **does the teacher's style matter?**

I fed 2,083 reasoning problems to both models via Ollama cloud, captured every thought, trained two separate Qwen3.5-4B students, and compared them.

Every step — including the mistakes — is in the [Dev Log](DEVLOG.md).

## Results

| Model | Training | GSM8K Accuracy | Format compliance | Avg thinking tokens |
|-------|----------|---------------|-------------------|---------------------|
| Base Qwen3.5-4B | None | TBD | 0% | 0 |
| GLM-5 distilled | SFT on GLM-5 traces | TBD | TBD | TBD |
| Kimi K2.5 distilled | SFT on Kimi traces | TBD | TBD | TBD |
| Combined | SFT on both mixed | TBD | TBD | TBD |

*Results updated after eval runs*

## How it works

| Step | What | Time |
|------|------|------|
| 1. Collect problems | GSM8K, MATH, ARC, HumanEval — 2,083 total | 5 min |
| 2. Generate traces | GLM-5 solves each with full `<think>` chain via Ollama cloud | ~8 hrs |
| 3. Filter | Keep correct answers with deep reasoning (>50 thinking tokens) | 15 min |
| 4. Format | Convert to chat format with `<think>`/`<answer>` tags, 80/10/10 split | 1 min |
| 5. Train | LoRA SFT via Tinker — two separate models, one per teacher | ~2 hrs each |
| 6. Evaluate | 4-way: base vs GLM-5 distilled vs Kimi distilled vs combined | 30 min |
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

| Dataset | Teacher | Format |
|---------|---------|--------|
| [bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) | GLM-5 | Raw (problem, thinking, response) |
| [bmeyer2025/glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft) | GLM-5 | SFT-ready (train/val/test) |
| [bmeyer2025/kimi-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces) | Kimi K2.5 | Raw (problem, thinking, response) |
| [bmeyer2025/kimi-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces-sft) | Kimi K2.5 | SFT-ready (train/val/test) |

## What I learned

*(Updated after training completes)*

## License

MIT
