# distillreasoning

<p align="center">
  <img src="images/triptych_hero.png" alt="Three-panel triptych: GLM-5 teacher robot with dense chalkboard, Kimi teacher robot with concise chalkboard, both teachers graduating the tiny Qwen student robot" width="700">
</p>

<p align="center">
  <strong>I borrowed reasoning from a 744B and a 1T model and taught it to a 4B model that runs on a laptop.</strong>
</p>

<p align="center">
  <em>2,083 problems. Two teachers (GLM-5, Kimi K2.5). Qwen3.5-4B as student. SFT then GRPO. One controlled experiment.</em>
</p>

<p align="center">
  <a href="DEVLOG.md">Dev Log</a> |
  <a href="https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces">GLM-5 Dataset</a> |
  <a href="https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces">Kimi Dataset</a>
</p>

---

## The idea

Frontier models reason differently than small models — they think out loud, backtrack, check their work. That reasoning process is the valuable thing. If you can capture it and train a small model to imitate it, the small model punches way above its weight.

But does it matter *which* frontier model you learn from? GLM-5 writes verbose, detailed reasoning chains (median 433 tokens). Kimi K2.5 is more concise and elegant (median 325 tokens). Same problems, same student — **does the teacher's style matter?**

I fed 2,083 reasoning problems to both models via Ollama cloud, captured every thought, filtered through 8 quality gates, trained three separate Qwen3.5-4B students, and compared them against models up to 7x their size.

Every step — including the mistakes — is in the [Dev Log](DEVLOG.md).

## The experiment

A key question in distillation is: does the teacher model matter? To test this, we picked two frontier reasoning models with very different styles:

| | GLM-5 | Kimi K2.5 |
|--|-------|-----------|
| **Parameters** | 744B (40B active) | ~1T (32B active) |
| **AIME 2026** | 92.7% | Strong |
| **Reasoning style** | Verbose, thorough (median 433 tokens) | Concise, elegant (median 325 tokens) |
| **Filter keep rate** | 83.7% (more accurate, but rambles) | 86.5% (more errors, but cleaner traces) |
| **License** | MIT | MIT-compatible |
| **Ollama tag** | `glm-5:cloud` | `kimi-k2.5:cloud` |

Both are free via [Ollama cloud tags](https://ollama.com/search?c=thinking) — the model runs on Ollama's servers, you just send API calls. MIT licensed means outputs can legally be used for training ([we verified this](DEVLOG.md#100-pm--licensing-check-can-we-use-glm-5-outputs-for-training)).

**The setup:** Same 2,083 problems → both teachers → filter → train separate Qwen3.5-4B students → compare. Plus a third student trained on both trace sets combined. This tells us whether verbose reasoning (GLM-5) or concise reasoning (Kimi) transfers better to a small 4B model.

## Benchmarks

Evaluated on 4 clean benchmarks (zero overlap with training data) + 5 hand-written trick questions. All benchmarks verified for contamination before scoring.

### Results

Full benchmark results coming — using [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (the industry standard used by HuggingFace Open LLM Leaderboard) on Colab Pro H100. All 10 models evaluated with identical methodology.

| Benchmark | Shots | Method | Task |
|-----------|-------|--------|------|
| GSM8K | 8-shot CoT | Generative | `gsm8k_cot` |
| MATH | 4-shot | Generative (`\boxed{}`) | `minerva_math` |
| ARC-Challenge | 25-shot | Log-likelihood | `arc_challenge` |
| GPQA Diamond | 0-shot | Log-likelihood | `gpqa_diamond` |
| MMLU-Pro | 5-shot | Log-likelihood | `mmlu_pro` |

**Preliminary GSM8K (zero-shot, Tinker eval):** Distillation showed a +35 point lift (37% base → 72% Kimi distilled) and the distilled 4B outperformed a raw Qwen3-8B. Concise Kimi traces transferred better than verbose GLM-5 traces. Full results with proper lm-eval methodology pending.

**Why lm-eval?** We initially wrote custom extraction code and got garbage: 0% on MATH (extraction only checked `\boxed{}`), 100% on ARC (regex too generous). ARC/GPQA/MMLU-Pro use log-likelihood scoring, not generation — our approach was fundamentally wrong. Details in the [Dev Log](DEVLOG.md#eval-extraction-disaster).

We also caught a contamination issue mid-project — 94% overlap on our first eval attempt. Details in the [Dev Log](DEVLOG.md#the-contamination-disaster).

## How it works

| Step | What |
|------|------|
| 1. Collect | GSM8K, MATH, ARC, HumanEval — 2,083 problems |
| 2. Generate | Both teachers solve all problems with full `<think>` traces via Ollama cloud (4 parallel workers each) |
| 3. Filter | 8 quality gates: correctness, length bounds, coherence, repetition, structured reasoning |
| 4. Format | Chat format with `<think>`/`<answer>` tags, stratified train/val split |
| 5. SFT | LoRA fine-tune 3 models on Tinker: GLM-5 traces, Kimi traces, combined |
| 6. Benchmark | Eval on 4 clean benchmarks + 5 trick questions, 8 models compared |
| 7. GRPO | Reinforcement learning on all 3 SFT models — reward correct answers |
| 8. Final eval | Full comparison on Colab Pro (fast local GPU inference) |
| 9. Export | Merge LoRA → push to HuggingFace → GGUF for Ollama |

## Data pipeline

### Trace generation

4,166 total traces generated (2,083 per teacher) using Ollama cloud with 4 parallel workers per model. ~7 hours each, running concurrently.

### 8-gate quality filter

| Gate | What it checks | GLM-5 dropped | Kimi dropped |
|------|---------------|---------------|--------------|
| 1. Non-empty | Thinking + response exist | 2 | 0 |
| 2. Language quality | No encoding artifacts | 0 | 0 |
| 3. Length bounds | 50-4000 thinking tokens | 101 | 5 |
| 4. Correctness | Answer matches expected | 160 | 235 |
| 5. Repetition | No degenerate loops | 0 | 0 |
| 6. Coherence | Thinking references problem | 22 | 25 |
| 7. Self-contradiction | Max 2 self-corrections | 0 | 0 |
| 8. Structured reasoning | Step indicators present | 51 | 0 |

**GLM-5:** 1,744/2,083 kept (83.7%) — more accurate but very verbose
**Kimi:** 1,802/2,083 kept (86.5%) — more errors but cleaner traces

### Training data

| Dataset | Train | Val |
|---------|-------|-----|
| GLM-5 traces | 1,572 | 172 |
| Kimi traces | 1,624 | 178 |
| Combined | 3,196 | 350 |

## Quick start

```bash
git clone https://github.com/brianmeyer/distillreasoning.git
cd distillreasoning
python3 -m venv venv && source venv/bin/activate
pip install ollama datasets huggingface_hub tinker tinker-cookbook

# 1. Download 2,083 problems
python scripts/download_problems.py

# 2. Generate reasoning traces (runs overnight, 4 parallel workers each)
python scripts/generate_traces.py         # GLM-5
python scripts/generate_traces_kimi.py    # Kimi K2.5

# 3. Filter + format
python scripts/filter_traces.py glm5
python scripts/filter_traces.py kimi
python scripts/format_for_sft.py glm5
python scripts/format_for_sft.py kimi

# 4. Upload to HuggingFace
HF_TOKEN=xxx python scripts/upload_dataset.py

# 5. Train on Tinker (3 models in parallel)
TINKER_API_KEY=xxx python scripts/train_tinker.py

# 6. Eval + merge + publish (Colab Pro notebook)
# Open notebooks/eval_and_publish.ipynb
```

Or just use the pre-built datasets:
```python
from datasets import load_dataset

# GLM-5 traces
ds = load_dataset("bmeyer2025/glm5-reasoning-traces-sft")

# Kimi traces
ds = load_dataset("bmeyer2025/kimi-reasoning-traces-sft")
```

## Project structure

```
distillreasoning/
├── scripts/
│   ├── download_problems.py       # Pull GSM8K, MATH, ARC, HumanEval from HF
│   ├── generate_traces.py         # GLM-5 → reasoning traces (4 parallel workers)
│   ├── generate_traces_kimi.py    # Kimi K2.5 → reasoning traces
│   ├── filter_traces.py           # 8-gate quality filter (accepts: glm5|kimi)
│   ├── format_for_sft.py          # Stratified chat format (accepts: glm5|kimi)
│   ├── upload_dataset.py          # Push 4 datasets to HuggingFace
│   ├── train_tinker.py            # LoRA SFT on Tinker (3 models in parallel)
│   ├── train_grpo.py              # GRPO RL on top of SFT checkpoints (Tinker)
│   └── eval_one.py                # Eval single model (Tinker API, deprecated)
├── notebooks/
│   ├── sft_training.ipynb         # Colab notebook for SFT (Unsloth, backup to Tinker)
│   └── eval_and_publish.ipynb     # Colab Pro H100: lm-eval benchmarks → merge → publish
├── cards/
│   ├── dataset_card.md            # HuggingFace dataset card
│   └── model_card.md              # HuggingFace model card
├── images/                        # Generated hero images (Gemini)
├── DEVLOG.md                      # Full build log — every step and mistake
└── README.md
```

## Datasets

| Dataset | Teacher | Format | Link |
|---------|---------|--------|------|
| glm5-reasoning-traces | GLM-5 | Raw (problem, thinking, response) | [HuggingFace](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) |
| glm5-reasoning-traces-sft | GLM-5 | SFT-ready (train/val) | [HuggingFace](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft) |
| kimi-reasoning-traces | Kimi K2.5 | Raw (problem, thinking, response) | [HuggingFace](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces) |
| kimi-reasoning-traces-sft | Kimi K2.5 | SFT-ready (train/val) | [HuggingFace](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces-sft) |

## What I learned

1. **Concise teachers beat verbose ones for small students.** Kimi's 325-token median traces produced a better 4B student than GLM-5's 433-token traces. A 4B model can't absorb 6,000 words of reasoning — it overwhelms the model's capacity.

2. **Distillation can beat 2x model size.** Our 4B distilled model (72.6% GSM8K) outperforms a raw Qwen3-8B (63.0%). Targeted training on reasoning traces is more effective than raw scale.

3. **Benchmark contamination is easy to miss.** Our first eval showed 75-80% accuracy — actually 94% data overlap with training. Always verify eval data is clean before celebrating.

4. **The teacher's reasoning style transfers, not just answers.** The distilled model reasons step-by-step, sets up variables, writes equations, and verifies answers — patterns learned from the teacher traces. The reasoning *skill* transferred, not just the format.

5. **More data isn't always better.** Combined traces (3,196) scored 71.3% vs Kimi-only (1,624) at 72.6%. The verbose GLM-5 traces in the combined set may have added noise rather than signal.

*(More findings after GRPO and final eval)*

## License

MIT
