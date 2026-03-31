# Dev Log: Distilling GLM-5 Reasoning into a 4B Model

**Goal:** Take reasoning traces from GLM-5 (a 744B parameter model) and teach a tiny Qwen3.5-4B model to reason like it. Zero budget. Full pipeline from data generation to published model.

**Why:** I wanted to understand the full distillation pipeline hands-on — not just read about it. How do you actually go from a frontier model's reasoning to a small model that can run on a laptop? After [building a tiny LLM from scratch](https://github.com/brianmeyer/tinyllm), the natural next question was: can you take what a massive model knows and compress it into something small?

**Cost:** $0. Ollama cloud tags for generation, Google Colab free tier for training.

**Deliverables:**
- [bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) (dataset)
- [bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled](https://huggingface.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled) (model)
- GGUF exports for local inference

---

## The Core Idea: Knowledge Distillation

Knowledge distillation is one of the most important ideas in modern ML. The concept: a large, expensive "teacher" model generates training data that a small, cheap "student" model learns from. The student doesn't learn from the raw data directly — it learns from the teacher's *behavior* on that data.

**Why does this work?** Think of it like an experienced teacher grading math homework. The raw problem "What is 15% of 240?" has one right answer (36). But the teacher's worked solution — "15% means 15/100, so multiply 240 by 0.15, which gives 36" — contains much richer information. The student learns the *reasoning process*, not just the answer.

In LLM terms, this means:
1. Feed problems to a frontier model (GLM-5, 744B parameters)
2. Capture its full chain-of-thought reasoning (not just the answer)
3. Train a small model (Qwen3.5-4B) to produce similar reasoning chains
4. The small model learns reasoning *patterns* that transfer to new problems

**The key insight:** Reasoning traces are a form of "dark knowledge" — information the teacher model has that isn't in the original training data. When GLM-5 works through a problem step by step, it's revealing its internal problem-solving strategy. That strategy is what we're distilling.

### Why GLM-5 as the Teacher?

Three frontier reasoning models are available free via Ollama's cloud tags right now:

| Model | Total Params | Active Params | AIME 2026 | GPQA-Diamond | Ollama Tag |
|-------|-------------|---------------|-----------|--------------|------------|
| GLM-5 | 744B | 40B | 92.7% | 86.0% | `glm-5:cloud` |
| Kimi K2.5 | 1T | 32B | Strong | Strong | `kimi-k2.5:cloud` |
| MiniMax M2.7 | ~230B | 10B | N/A | N/A | `minimax-m2.7:cloud` |

GLM-5 wins for reasoning distillation because:
1. **Highest math/reasoning benchmarks** — 92.7% on AIME 2026 is exceptional
2. **Full `<think>` traces exposed** — unlike some providers that summarize or hide the reasoning chain, GLM-5 gives you everything
3. **MIT licensed, open weights** — no ambiguity about using outputs for training
4. **Free via Ollama cloud tag** — the model runs on Ollama's servers, you just send API calls

MiniMax M2.7 is more optimized for agentic/coding workflows — its traces tend to be more streamlined, which is the *opposite* of what you want for distillation (richer reasoning chains = more for the student to learn from).

### Why Qwen3.5-4B as the Student?

At 4 billion parameters, this model is small enough to:
- Fine-tune on a free Colab T4 GPU (16GB VRAM) using 4-bit quantization
- Run inference on basically any modern GPU or even CPU
- Export to GGUF for local use via Ollama/llama.cpp

But it's large enough to actually learn non-trivial reasoning patterns. Below ~1B parameters, models struggle to maintain coherent multi-step reasoning chains.

### The Training Pipeline

```
Problems (GSM8K, MATH, ARC, HumanEval)
    → GLM-5 generates reasoning traces via Ollama cloud
    → Filter for quality (correct answers, sufficient depth)
    → Format into chat template with <think>/<answer> tags
    → SFT with Unsloth QLoRA (train only on assistant responses)
    → Optional: GRPO reinforcement learning for correctness
    → Evaluate: base vs SFT vs GRPO
    → Export: HuggingFace + GGUF
```

**SFT (Supervised Fine-Tuning)** teaches the model to produce reasoning traces that *look like* GLM-5's. It's learning the style and structure of step-by-step reasoning.

**GRPO (Group Relative Policy Optimization)** goes further — it rewards the model for traces that actually *lead to correct answers*. SFT is learning to write proofs by copying examples. GRPO is learning to write proofs by getting graded on them.

---

## Phase 1: Dataset Generation — March 31, 2026

### 10:31 AM — Project Setup

Empty directory. Checked prerequisites:

```
$ which ollama && ollama --version
/opt/homebrew/bin/ollama
ollama version is 0.19.0

$ python3 --version
Python 3.12.12
```

Ollama installed, Python ready. Created the project:
```bash
mkdir distillreasoning && cd distillreasoning
git init && git branch -m main
mkdir -p scripts data
```

### 10:35 AM — First Mistake: pip install on macOS

Tried to install packages globally:
```bash
$ pip3 install ollama datasets huggingface_hub
error: externally-managed-environment
× This environment is externally managed
```

**PEP 668** strikes again. macOS now blocks system-wide pip installs to protect the system Python. I hit this on tinyllm too and apparently didn't learn. Always venv first.

```bash
python3 -m venv venv
./venv/bin/pip install ollama datasets huggingface_hub
```

Installed: ollama 0.6.1, datasets 4.8.4, huggingface_hub 1.8.0.

### 10:38 AM — Testing GLM-5 Cloud

First, pull the cloud model tag:
```bash
ollama pull glm-5:cloud
```

This downloads a tiny ~323 byte manifest — the actual 744B model runs on Ollama's infrastructure. You're essentially getting an API endpoint for free.

Then tested with the Python library:

```python
import ollama
response = ollama.chat(
    model='glm-5:cloud',
    messages=[{'role': 'user', 'content': 'What is 7 * 8 + 3? Think step by step.'}],
    think=True  # This enables the thinking trace
)
```

**The `think=True` flag is critical.** Without it, you just get the final answer. With it, you get the full reasoning chain in a separate `response.message.thinking` field.

The thinking trace for "7 * 8 + 3":

> "Analyze the user's request... Identify the operation... Recall order of operations (PEMDAS/BODMAS)... Multiplication first: 7 × 8 = 56... Then addition: 56 + 3 = 59"

468 words of reasoning for a trivial arithmetic problem. That verbosity is actually what we want — richer traces give the student model more to learn from.

**Gotcha:** I initially ran `source venv/bin/activate && python3 -c "import ollama..."` but the Python subprocess couldn't find the ollama module. The activate script sets environment variables for the *shell*, but when Claude Code spawns a subprocess, those don't always propagate. Fix: use `./venv/bin/python` directly to ensure the right Python interpreter runs.

### 10:45 AM — Downloading Problem Sets

For distillation to work well, you need diverse problems that exercise different reasoning skills. I pulled from four sources:

1. **GSM8K** — Grade school math word problems. Clear numeric answers. The gold standard for math reasoning evaluation. Example: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"

2. **MATH** — Competition-level math (AMC, AIME difficulty). Much harder than GSM8K. Includes algebra, geometry, number theory, combinatorics. Answers often in LaTeX `\boxed{}` format.

3. **ARC-Challenge** — Science reasoning multiple choice. Tests knowledge + inference. Example: "Which factor most accurately describes the





cause of pollution in a river that__(continues)..."

4. **HumanEval** — Coding problems where you write Python functions. Tests logical/algorithmic reasoning. Every problem has a verifiable correct solution.

### 10:50 AM — Second Mistake: The MATH Dataset Doesn't Exist (Anymore)

The guide I wrote referenced `lighteval/MATH`. First attempt:
```python
load_dataset("lighteval/MATH", "all", split="test")
# DatasetNotFoundError: Dataset 'lighteval/MATH' doesn't exist on the Hub
```

OK, maybe the original source:
```python
load_dataset("hendrycks/competition_math", split="test")
# FileNotFoundError: Couldn't find 'hendrycks/competition_math' on the Hub
```

Two strikes. The original MATH dataset by Dan Hendrycks has been taken down or reorganized on HuggingFace. **This is a real problem with ML workflows** — datasets move, get renamed, get restricted, or disappear entirely. The URL you wrote in your notes three months ago may not work today.

Had to search HuggingFace Hub for a mirror. Found one:
```python
load_dataset("SuperSecureHuman/competition_math_hf_dataset", split="test")
# ✅ Works! 5,000 test problems
```

Same data, different uploader. Problem solved, but it burned 5 minutes and is a reminder: **always verify dataset availability before building a pipeline around it.**

### 10:55 AM — 2,083 Problems Downloaded

More than the 1,200 target. The generate script will stratified-sample down.

| Source | Count | What it tests |
|--------|-------|---------------|
| GSM8K | 1,319 | Math word problems (numeric answers) |
| MATH | 200 | Competition math (LaTeX answers) |
| ARC-Challenge | 400 | Science reasoning (multiple choice) |
| HumanEval | 164 | Coding (Python functions) |
| **Total** | **2,083** | |

All saved to `data/problems.jsonl` as structured records with `id`, `source`, `problem`, and `expected_answer` fields.

### 11:00 AM — The Big One: Generating Reasoning Traces

This is where the actual distillation data gets created. The script (`scripts/generate_traces.py`) does:

1. **Stratified sampling** — takes ~1,200 problems proportional to each source's representation
2. **System prompt** — "You are a reasoning expert. Think through each problem step by step in detail before giving your final answer. Show all your work."
3. **Incremental saves** — writes each completed trace to JSONL immediately, so if the script crashes after 500 traces, you keep those 500
4. **Resume capability** — on restart, it checks which IDs are already in the output file and skips them
5. **Rate limiting** — 2-second delay between requests to avoid throttling the free cloud API
6. **Retry logic** — 3 attempts with exponential backoff (5s, 10s, 15s)

Kicked it off and watched the first results roll in:

```
[1/1198]  gsm8k_862 (gsm8k)...    OK (thinking: 468 words)
[2/1198]  gsm8k_376 (gsm8k)...    OK (thinking: 528 words)
[3/1198]  arc_181 (arc)...         OK (thinking: 554 words)
[4/1198]  gsm8k_495 (gsm8k)...    OK (thinking: 268 words)
[5/1198]  gsm8k_520 (gsm8k)...    OK (thinking: 476 words)
[6/1198]  gsm8k_699 (gsm8k)...    OK (thinking: 222 words)
[7/1198]  gsm8k_733 (gsm8k)...    OK (thinking: 5,945 words)
[8/1198]  gsm8k_722 (gsm8k)...    OK (thinking: 299 words)
[10/1198] humaneval_119 (humaneval)... OK (thinking: 2,254 words)
```

**The variance is fascinating.** Problem 6 got 222 words of thinking (probably a straightforward calculation). Problem 7 got **5,945 words** — GLM-5 went deep on that one, probably involving multiple solution attempts or a complex proof. HumanEval problems average much longer traces because the model reasons about algorithm design, edge cases, and code structure.

**Estimated completion time:** ~10 seconds per problem (API call + 2s delay) × 1,198 problems = **~3.3 hours**

Moved to background: `nohup ./venv/bin/python scripts/generate_traces.py > data/generation.log 2>&1 &`

### 11:15 AM — Building Pipeline Scripts While Waiting

With trace generation running for the next 3+ hours, I built out every other script we'll need:

#### Filter Script (`scripts/filter_traces.py`)

Not all traces will be usable. The filter checks:

- **GSM8K traces:** Extract the last number from the response, compare to expected answer. Drop incorrect ones. This is strict — if the model's reasoning was beautiful but arrived at the wrong number, we drop it. We only want to teach the student correct reasoning.

- **ARC traces:** Check if the expected multiple choice letter appears in the response. Look for patterns like "The answer is B" or "(B)" or just the letter at the end. Harder to verify automatically than numeric answers.

- **MATH traces:** These have complex LaTeX answers (`\boxed{-36}`). Hard to auto-verify, so we just check that the response contains *some* answer (either `\boxed{}` or the word "answer").

- **HumanEval traces:** Code verification is complex (you'd need to run the code). Keep all of them and rely on manual spot-checking.

- **All traces:** Drop any with thinking under 50 tokens (model didn't engage deeply). Drop any with >50% sentence repetition (model got stuck in a loop).

**Target:** ~800-1,000 quality traces from ~1,200 inputs.

#### Format Script (`scripts/format_for_sft.py`)

Converts filtered traces into the chat format Unsloth/TRL expect:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful reasoning assistant..."},
    {"role": "user", "content": "<the problem>"},
    {"role": "assistant", "content": "<think>\n...\n</think>\n\n<answer>\n...\n</answer>"}
  ]
}
```

The `<think>`/`<answer>` tags are important — they give the model a structured format to learn. During inference, the model will learn to put its reasoning in `<think>` and its final answer in `<answer>`. This makes it easy to parse both parts.

Split: 90% train, 10% test.

#### Colab Training Notebook (`notebooks/sft_training.ipynb`)

The key training decisions:

- **QLoRA (4-bit quantization + LoRA):** The base Qwen3.5-4B model is loaded in 4-bit precision (~2GB VRAM) with LoRA adapters added on top. Only the LoRA weights get trained — the base model stays frozen. This is how you fine-tune a 4B model on a free T4 (16GB VRAM).

- **LoRA rank 32:** Higher rank = more trainable parameters = more capacity to learn reasoning patterns. Rank 32 on 7 target modules (q, k, v, o, gate, up, down projections) gives us plenty of capacity.

- **`train_on_responses_only`:** This is critical for distillation. The loss is only computed on the assistant's response (the reasoning + answer), NOT on the system prompt or user question. Without this, the model wastes capacity "learning" to reproduce the problem statement, which it doesn't need to do. With it, every gradient update is focused on learning reasoning.

- **3 epochs, effective batch size 8:** Standard SFT recipe. Cosine LR schedule with 10% warmup.

### 11:30 AM — Third Mistake: Git Identity Not Configured

Created the GitHub repo fine:
```bash
gh repo create distillreasoning --public
# ✅ https://github.com/brianmeyer/distillreasoning
```

But the commit failed:
```
Author identity unknown
*** Please tell me who you are.
```

This machine doesn't have git user.name/user.email set. Minor but annoying — need to configure this before I can push.

### Status Check — 11:45 AM

| What | Status |
|------|--------|
| Project structure | ✅ Done |
| Dependencies | ✅ Installed |
| GLM-5 cloud | ✅ Tested and working |
| Problem sets | ✅ 2,083 downloaded |
| Trace generation | 🔄 Running (~25/1198) |
| Filter/format scripts | ✅ Written |
| Colab notebook | ✅ Written |
| GitHub repo | ⚠️ Created, commit pending (git identity) |
| HuggingFace dataset | ⏳ After traces finish |
| SFT training | ⏳ After dataset uploaded |

**The bottleneck is trace generation.** Everything else is ready and waiting. ETA ~2:30 PM for traces to complete.

---

### 11:50 AM — Scaling Up: Going for All 2,083 Problems

Original target was ~1,200 problems. Bumped to 1,500, then decided to just use everything. We have 2,083 problems downloaded — no reason to throw any away. More diverse training data is almost always better for distillation. The filter step will cull the bad ones anyway.

Updated `MAX_PROBLEMS = 9999` in the generate script (effectively "use all"). Killed and restarted the generation process. Because of the incremental save + resume logic, the 39 traces already done were preserved.

**Updated ETA:** All 2,083 problems × ~14 seconds each = **~8 hours** of generation. Kicked off at ~11:53 AM, expect completion around **8 PM**.

Updated the format script to do a proper **80/10/10 train/validation/test split** (was 90/10 before). The three-way split matters:
- **Train (80%)** — what the model learns from
- **Validation (10%)** — monitor loss during training, catch overfitting early
- **Test (10%)** — completely held out, never seen during training, used for final eval

### 12:15 PM — Switching from Colab to Tinker

Originally planned to use Google Colab free tier (T4 GPU) for training with Unsloth. After looking at the options again, switching to **[Tinker](https://thinkingmachines.ai/tinker/)** — the fine-tuning API from Mira Murati's Thinking Machines Lab.

**Why Tinker over Colab:**

| | Colab Free | Tinker |
|--|-----------|--------|
| GPU | T4 (16GB) | Managed cloud |
| Timeout | ~4-6 hrs | None |
| Checkpoints | Dies on disconnect (learned from tinyllm) | Persistent |
| GRPO/RL | Manual setup | Native support |
| Cost | Free | ~$6-7 for our run ($150 credit available) |

The tinyllm project got burned by Colab disconnecting after 3+ hours of training — losing all checkpoints. Tinker avoids that entirely. With $150 in credits and Qwen3.5-4B costing $0.67/million training tokens, we can run the full SFT (~10M tokens) plus multiple GRPO experiments for well under $50.

**Key question: is LoRA-only a problem?** Tinker only supports LoRA, not full fine-tuning. But this is exactly what the original Unsloth/Colab plan used too (QLoRA). At 4B parameters, LoRA rank 32 across all attention and MLP layers gives plenty of capacity for distillation. The LoRA weights get merged back into the base model at the end — the final exported model is a full standalone model, indistinguishable from a fully fine-tuned one.

**The data format was already compatible.** Tinker expects `{"messages": [...]}` JSONL — exactly what our pipeline already produces. Zero reformatting needed.

### 12:30 PM — Installing and Authenticating Tinker

```bash
pip install tinker tinker-cookbook
tinker version
# tinker 0.16.1

tinker run list
# No training runs found (clean account)
```

API key authenticated. Account is fresh, ready to go.

Tested that Qwen3.5-4B is supported and the renderer is available:
```python
from tinker_cookbook import model_info
renderer = model_info.get_recommended_renderer_name("Qwen/Qwen3.5-4B")
# qwen3_5 ✅
```

### 12:45 PM — Writing the Tinker Training Script

Read the [Tinker docs](https://tinker-docs.thinkingmachines.ai) carefully before writing the script. Key things I learned:

**The Tinker API is lower-level than Unsloth/HuggingFace Trainer.** Instead of a `trainer.train()` call, you manually loop and call:
1. `training_client.forward_backward(batch, "cross_entropy")` — compute gradients
2. `training_client.optim_step(AdamParams(lr))` — update weights

This is more verbose but also more transparent. You can see exactly what's happening at each step.

**Weight masking via the renderer.** The key call is:
```python
model_input, weights = renderer.build_supervised_example(messages)
```
This applies the model's chat template AND automatically sets `weights=0` for system/user tokens and `weights=1` for assistant tokens. So the model only learns from the reasoning traces, not from the questions themselves. This is the Tinker equivalent of Unsloth's `train_on_responses_only`.

**Learning rate formula.** Tinker has a specific formula for LoRA learning rates:
`LR(m) = lr_base * M_LoRA * (2000/H_m)^P_m`

Rather than guess, used their helper:
```python
from tinker_cookbook.hyperparam_utils import get_lr
lr = get_lr("Qwen/Qwen3.5-4B")
```

**Recommended settings per docs:**
- Batch size: 128
- Min training steps: 100 (we'll do ~1,000+)
- LoRA rank: 32 (default)

### Status — 1:00 PM

| What | Status |
|------|--------|
| Trace generation | 🔄 ~80/2083 running (ETA ~8 PM) |
| Tinker SDK | ✅ Installed, authenticated |
| Tinker training script | ✅ Written (`scripts/train_tinker.py`) |
| Colab notebook | ✅ Still exists as backup |
| Filter/format scripts | ✅ Ready to run after traces finish |

**Waiting on:** Trace generation. Everything else is ready. When generation finishes tonight:
1. `python scripts/filter_traces.py`
2. `python scripts/format_for_sft.py`
3. `HF_TOKEN=xxx python scripts/upload_dataset.py`
4. `TINKER_API_KEY=xxx python scripts/train_tinker.py`

### 1:15 PM — Publishing Two Datasets to HuggingFace

Decided to publish the data in two forms rather than one:

**`bmeyer2025/glm5-reasoning-traces`** — The raw traces. Each row has:
- `id` — problem identifier
- `source` — gsm8k / math / arc / humaneval
- `problem` — the original question
- `expected_answer` — ground truth answer
- `thinking` — GLM-5's full `<think>` block (the gold)
- `response` — GLM-5's final answer

This is the most reusable form. Anyone who wants to distill into a different model, use a different chat template, or build a different pipeline can start from the raw traces.

**`bmeyer2025/glm5-reasoning-traces-sft`** — The formatted version. Same data but already converted to `{"messages": [...]}` format with `<think>`/`<answer>` tags in the assistant turn, split 80/10/10 into train/validation/test. Plug straight into any HuggingFace-compatible trainer.

Updated `upload_dataset.py` to push both repos in a single run.

---

## Errors and Lessons So Far

| What happened | Why | What we learned |
|--------------|-----|----------------|
| `pip install` failed on macOS | PEP 668 blocks system-wide installs | Always create a venv first on macOS. Hit this on tinyllm too — apparently I don't learn. |
| `lighteval/MATH` dataset not found | Dataset removed/reorganized on HuggingFace | Popular datasets move around. Verify availability before building a pipeline. Found mirror at `SuperSecureHuman/competition_math_hf_dataset`. |
| `hendrycks/competition_math` also not found | Same issue, original source gone too | Tried two paths before finding a working one. Cost 5 minutes. |
| `source venv/bin/activate` didn't propagate to subprocess | Shell env vars don't always transfer to child processes | Use `./venv/bin/python` directly instead of relying on activate |
| Git commit failed — no identity | Fresh machine, no git config | Need to set user.name and user.email before first commit |

---

## Key Concepts Glossary

**Knowledge Distillation**: Training a small "student" model on the outputs of a large "teacher" model. The student learns from the teacher's behavior rather than from raw data, capturing "dark knowledge" that isn't in the original training set.

**Reasoning Traces**: The step-by-step thinking process a model goes through before arriving at an answer. In GLM-5, these are exposed via `<think>` blocks. Richer traces = more for the student to learn from.

**QLoRA (Quantized Low-Rank Adaptation)**: Load the base model in 4-bit precision (saves ~75% VRAM), then add small trainable LoRA adapters. Only the adapters get trained — the base model stays frozen. This lets you fine-tune large models on consumer GPUs.

**SFT (Supervised Fine-Tuning)**: Train the model to produce outputs that match examples in the dataset. For distillation, the "examples" are the teacher's reasoning traces. The model learns to imitate the teacher's reasoning style.

**GRPO (Group Relative Policy Optimization)**: A reinforcement learning method that generates multiple responses, scores them with reward functions, and updates the model to prefer higher-scoring responses. Unlike SFT which just imitates, GRPO optimizes for actual correctness.

**train_on_responses_only**: A training configuration that masks the loss on system/user messages, only computing gradients on the assistant's response. Essential for distillation — you want the model to learn reasoning, not to learn to repeat the question.

**Ollama Cloud Tags**: Free API endpoints for frontier models. The model runs on Ollama's servers, you get a local-feeling API. Currently available: GLM-5, Kimi K2.5, MiniMax M2.7. No API key needed, no rate limits (beyond reasonable use), no cost.

**GGUF**: A file format for running LLMs locally via llama.cpp/Ollama. Supports various quantization levels (q4_k_m for speed, q8_0 for quality). The end goal — a model file you can run on your laptop.
