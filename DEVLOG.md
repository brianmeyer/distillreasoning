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

### 1:30 PM — Images and Repo Presentation

Generated three images in Gemini matching the tinyllm visual style (dark navy backgrounds, glowing blue tech elements, warm/cold contrast):

- **`classroom_hero.png`** → GitHub README header. GLM-5 teacher robot at a chalkboard showing `<think>` reasoning chains, tiny Qwen3.5-4B student robot taking notes. Labels: "GLM-5 (744B)" and "QWEN3.5-4B". Apple on the desk.
- **`distillation_brain.png`** → HuggingFace dataset card. Large glowing neural network brain streaming equations and reasoning chains into a compact glowing crystal. Captures the "big model → small model" concept visually.
- **`distillation_apparatus.png`** → HuggingFace model card. Chemistry distillation apparatus with math equations flowing through glass tubes, condensing into a flask labeled "PURE DISTILLED REASONING".

One image per destination — GitHub gets the classroom, dataset card gets the brain, model card gets the apparatus.

GitHub repo description updated to: *"Borrow reasoning from a 744B model. Teach it to a 4B model. Run it on your laptop. Zero cost."*

Also created `cards/dataset_card.md` and `cards/model_card.md` — full HuggingFace repo cards with metadata frontmatter, usage examples, eval table placeholder. Both get uploaded automatically by `upload_dataset.py`.

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

---

### 12:30 PM — Parallelizing Trace Generation (3.5x Speedup)

The sequential generator was averaging ~2 traces/minute — each request takes 5-15 seconds for the API response, plus the 2-second rate-limiting delay we added to be safe. At that rate: **21 hours** for 2,083 problems. Overnight plus most of tomorrow.

Tested whether Ollama cloud handles concurrent requests:
```python
# Sequential: 12.9s for 4 problems
# Parallel (4 workers): 8.9s for 4 problems → 1.4x on easy problems
```

It works! Rewrote `generate_traces.py` with:
- **4 parallel workers** via `ThreadPoolExecutor`
- **0.5s delay** between submitting requests (down from 2s)
- Thread-safe file writing with locks
- Per-trace rate counter (traces/hour)

Tested on simple problems first to check for rate limiting. Result: **zero failures, zero empty traces, zero rate limit errors.** Ollama cloud handles 4 concurrent requests fine.

New rate: **~5 traces/minute** (up from ~2/min). ETA dropped from **21 hours → ~6.5 hours**. Should finish by ~7 PM.

One gotcha: nohup buffers Python stdout even with `-u` when output comes from thread pools. The traces were being saved correctly to the JSONL file, but the console log wasn't updating. Fixed by using `python -u` (unbuffered) — but the thread-to-main-thread print flow still buffers. Not a real problem since we verify progress by counting lines in the output file, not by reading the log. This is the same "Python output buffering" lesson from tinyllm.

### 1:00 PM — Licensing Check: Can We Use GLM-5 Outputs for Training?

Important question to nail down before publishing anything. Short answer: **yes, unambiguously.**

GLM-5 is MIT licensed — the most permissive open-source license. Released by Zhipu AI on Feb 13, 2026. MIT permits unrestricted commercial use, modification, and redistribution with no carve-outs for model outputs or distillation.

This matters because **not all models allow this**. OpenAI and Anthropic's terms of service explicitly prohibit using outputs to train competing models. Meta's Llama community license has restrictions on competitive use. If you're doing distillation, the teacher model's license is the first thing to check.

GLM-5, DeepSeek R1, and a few others are genuinely MIT — outputs included. This is one of the key reasons we chose GLM-5 as the teacher.

### 1:30 PM — Adding Kimi K2.5 as a Second Teacher

Decided to make this a **controlled experiment**: same problems, same student model, different teacher. If we train two separate Qwen3.5-4B models — one on GLM-5 traces, one on Kimi K2.5 traces — we can answer: **does the teacher matter, or does any frontier model work?**

The four-way comparison:

| Model | Training | Question it answers |
|-------|----------|---------------------|
| Base Qwen3.5-4B | None | How bad is the baseline? |
| GLM-5 distilled | SFT on GLM-5 traces | How does the best math reasoner do? |
| Kimi K2.5 distilled | SFT on Kimi traces | Does a different reasoning style matter? |
| Combined | SFT on both mixed | Is more diverse data better? |

**Why this is interesting:** GLM-5 generates very verbose traces (468+ words for simple math). Kimi K2.5 is more concise (113 words for the same problem). If both produce similar distilled models, the style doesn't matter — it's the *correctness* of the reasoning that transfers. If GLM-5's verbose style wins, it suggests that richer training signal helps. If Kimi wins despite being more concise, it suggests the student prefers cleaner examples.

Kimi K2.5 is also free via Ollama cloud and MIT-compatible, so there's no cost or licensing issue.

**Concurrency test:** Ran GLM-5 and Kimi requests simultaneously:
```
GLM-5: 23.7s ✅
Kimi:  23.7s ✅
GLM-5: 27.0s ✅
Kimi:  26.9s ✅
```
Both work concurrently but slower — responses went from ~5-10s to ~25s. Both models running at the same time puts more load on Ollama's infrastructure. Set Kimi to 2 workers (vs GLM-5's 4) to be conservative.

GLM-5 generation rate dropped from ~6/min to ~3/min with Kimi running alongside. Kimi doing ~2/min. Combined throughput: ~5/min. Net effect is about the same total rate but generating two datasets simultaneously.

**Renamed files:**
- `traces_raw.jsonl` → `traces_raw_glm5.jsonl` (avoid confusion)
- `traces_raw_kimi.jsonl` (new)
- All downstream scripts now accept `[glm5|kimi]` as an argument

**New HuggingFace dataset plan:**
- `bmeyer2025/glm5-reasoning-traces` — raw GLM-5 traces
- `bmeyer2025/glm5-reasoning-traces-sft` — formatted GLM-5 traces
- `bmeyer2025/kimi-reasoning-traces` — raw Kimi traces
- `bmeyer2025/kimi-reasoning-traces-sft` — formatted Kimi traces

*Both generators running. GLM-5: 227/2083, Kimi: 6/2083. ETA: both ~6-8 hours.*

### 1:45 PM — Expanding the Experiment: Two Students, Three Training Stages

The project scope grew from "distill one model" to a controlled experiment. Here's the full design:

**Two student models:**
- Qwen3.5-4B — sweet spot for capacity
- Qwen3.5-2B — tests how small you can go before distillation stops working

**Three teacher configs:**
- GLM-5 only (verbose reasoning)
- Kimi K2.5 only (concise reasoning)
- Combined (both trace sets merged)

**Three training stages, benchmarked at each:**

| Stage | What it does | Method |
|-------|-------------|--------|
| **Base** | No training — the control | — |
| **SFT** | Learn to produce reasoning traces | LoRA fine-tuning on teacher traces |
| **SFT → GRPO** | Reinforce correct answers | RL with reward functions on top of SFT |

LoRA is used in both SFT and GRPO — it's the adapter method, not a separate training type. SFT teaches the student to *reason like the teacher* (style + structure). GRPO teaches it to *reason correctly* (reward right answers, penalize degenerate output).

**The full eval matrix:**

| Student | Teacher | Base | After SFT | After GRPO |
|---------|---------|------|-----------|------------|
| 4B | GLM-5 | eval | eval | eval (if top SFT) |
| 4B | Kimi | eval | eval | eval (if top SFT) |
| 4B | Combined | eval | eval | eval (if top SFT) |
| 2B | GLM-5 | eval | eval | eval (if top SFT) |
| 2B | Kimi | eval | eval | eval (if top SFT) |
| 2B | Combined | eval | eval | eval (if top SFT) |

14 eval points total. GRPO only on the top 2-3 SFT performers (not all 6 — diminishing returns on the weaker ones).

**GRPO reward functions:**
- `correctness_reward(1.0)` — did the model get the right numeric answer?
- `format_reward(0.2)` — did it use proper `<think>`/`<answer>` tags?
- `repetition_penalty(-0.3)` — is the model stuck in a loop?

**Questions this answers:**
1. Does the teacher model matter? (GLM-5 vs Kimi vs combined)
2. Does student size matter? (4B vs 2B with the same teacher)
3. Does verbose or concise reasoning transfer better?
4. Does GRPO meaningfully improve on SFT?
5. At what model size does distillation break down?

Created a Linear project to track all of this: [Distill Reasoning](https://linear.app/recallforge/project/distill-reasoning-06471c1b227f) with 9 issues covering every phase from trace generation through final publish.

### 2:00 PM — Overhauling the Filter Pipeline

Took a hard look at the filter script and realized it was pretty weak. The original version only checked:
- Minimum thinking length (50 tokens)
- Sentence-level repetition
- GSM8K numeric answer matching
- ARC letter matching
- MATH: just checking for the word "answer" (way too loose)
- HumanEval: no verification at all

**The problem:** dataset quality is everything in distillation. A smaller, clean dataset beats a larger noisy one. If we feed the student model traces where GLM-5 got the answer wrong, we're teaching it to reason *incorrectly*. If we include traces where the model rambles for 6,000 words, we're wasting training tokens on noise.

Rewrote with **8 quality gates** applied in order:

| Gate | What it checks |
|------|---------------|
| 1. Non-empty | Thinking and response must exist |
| 2. Language quality | No encoding artifacts, garbled text, excessive non-ASCII |
| 3. Length bounds | Min 50, max 4,000 thinking tokens. Min 5 response tokens |
| 4. Correctness | Source-specific answer verification (numeric, multiple choice, boxed LaTeX, code) |
| 5. Repetition | Sentence deduplication + trigram frequency analysis |
| 6. Coherence | Does the thinking actually reference the problem's content? |
| 7. Self-contradiction | Max 2 self-corrections (one is fine, three = confused) |
| 8. Structured reasoning | Must contain step indicators (step 1, first, therefore, etc.) |

**Test run on partial data (both generators still running):**

| Metric | GLM-5 (1,488 traces) | Kimi (1,455 traces) |
|--------|---------------------|---------------------|
| **Keep rate** | **83.9%** | **86.3%** |
| Wrong answers (GSM8K) | 84 (8.9%) | 135 (14.6%) |
| Too long (>4000 tok) | 65 | 14 |
| No reasoning structure | 33 | 0 |
| Incoherent | 15 | 18 |
| Median thinking tokens | 418 | 320 |
| Mean thinking tokens | 661 | 531 |

**Early observations already visible:**
1. **GLM-5 is more accurate on math** — 8.9% wrong answer rate vs Kimi's 14.6%. Matches GLM-5's higher AIME benchmark.
2. **Kimi is more concise** — median 320 vs 418 thinking tokens, and far fewer "too long" drops (14 vs 65). GLM-5 tends to over-explain.
3. **Kimi is better on ARC/HumanEval** — 98.6% and 98.2% keep rates vs GLM-5's 88.3% and 91.8%. Concise reasoning works well for multiple choice and code.
4. **MATH is hard for both** — GLM-5 54.5%, Kimi 64.8%. Competition math pushes both teachers.

These are preliminary numbers on partial data (~70% generated). Final numbers will change but the patterns are clear.

### 2:30 PM — Stratified Splitting

Caught a problem with the format script — it was doing a naive random shuffle then splitting 80/10/10. That means the test set could end up with zero HumanEval or zero MATH problems by luck of the draw. Bad for evaluation.

Rewrote `format_for_sft.py` with **stratified splitting**: each domain (gsm8k, math, arc, humaneval) gets proportional representation in every split. Every domain is guaranteed to have samples in train, val, AND test. No domain disappears.

Verified on GLM-5 partial data:

| Source | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| gsm8k | 814 | 651 | 81 | 82 |
| arc | 256 | 204 | 25 | 27 |
| humaneval | 101 | 80 | 10 | 11 |
| math | 78 | 62 | 7 | 9 |

### 2:45 PM — Spot Check: Are the Answers Actually Right?

Good filters mean nothing if we don't verify the output. Did a random spot check — 2 samples per source per teacher, 16 traces total. Checked model's final answer against expected answer.

**Result: 16/16 correct.** Every trace in the filtered dataset arrived at the right answer with proper reasoning chains. The 8-gate filter is working — wrong answers are being caught and dropped before they can poison the training data.

Some examples of what survived filtering:
- GLM-5 on gsm8k_853 (hotel rooms): 968 words of thinking, correct answer of 15 hours
- Kimi on math_3786 (algebraic equation): 160 words of thinking, correct answer of x=-5 with verification step
- Both teachers on ARC: proper elimination reasoning, correct letter choices

The contrast between teachers is visible even in spot checks. GLM-5 wrote 968 words for a word problem. Kimi wrote 251 words for a similar difficulty problem. Same correct answer, very different reasoning depth. This is exactly the variable we're testing.

### Status — 3:00 PM

Both generators past 70%, running side by side. Everything downstream is built, tested, and ready. Waiting on generation to complete (~2-3 hours).

| Component | Status |
|-----------|--------|
| Trace generation (GLM-5) | 🔄 1,514/2,083 (73%) |
| Trace generation (Kimi) | 🔄 1,488/2,083 (71%) |
| 8-gate filter | ✅ Tested, 83-86% keep rate |
| Stratified formatter | ✅ Tested, domains balanced |
| Spot check | ✅ 16/16 correct |
| Upload script | ✅ Ready (4 HF datasets) |
| Tinker training | ✅ Script ready, API verified |
| GitHub repo | ✅ Current |
| Linear tracking | ✅ 9 issues (REC-216 to REC-224) |
| Devlog | ✅ Up to date |

---

### Trace Generation Complete!

Both generators finished. Final stats:

| Teacher | Traces | Time |
|---------|--------|------|
| GLM-5 | 2,083 | ~7 hours |
| Kimi K2.5 | 2,083 | ~7 hours |
| **Total** | **4,166** | (ran concurrently) |

Zero failures on either generator. The parallel 4+4 worker setup with 0.5s stagger worked with no rate limiting from Ollama cloud.

### Evaluation Benchmarks — Following DeepSeek-R1's Approach

Before running the pipeline, rewrote the evaluation script. Our original eval was just GSM8K accuracy + format compliance — too narrow. Looked at how [DeepSeek evaluated their R1-Distill models](https://arxiv.org/html/2501.12948v1):

- **5 benchmarks** (AIME, MATH-500, GPQA Diamond, LiveCodeBench, Codeforces)
- **Zero-shot** (few-shot actually hurts reasoning models)
- **Pass@1** with temperature 0.6, top-p 0.95
- **Consensus@64** (majority vote) for harder benchmarks

Our eval now covers **3 benchmarks + qualitative comparison:**

| Benchmark | N | What it tests | Metric |
|-----------|---|---------------|--------|
| GSM8K | 100 | Grade school math | Numeric answer match |
| MATH | 100 | Competition math | Boxed answer match |
| ARC-Challenge | 100 | Science reasoning | Letter answer match |
| Trick questions | 5 | Reasoning quality | Qualitative side-by-side |

Each benchmark reports: accuracy, format compliance, avg thinking tokens, avg response time.

The 5 trick questions (bat & ball, sheep, widgets, etc.) are for the devlog/article — same problems, all models, side by side. These are the ones where reasoning matters most because the intuitive answer is wrong.

---

## Phase 2: Filtering — Final Results

Both trace sets complete. Ran `scripts/filter_traces.py` through all 8 quality gates.

### GLM-5: 1,744/2,083 kept (83.7%)

| Source | Total | Kept | Keep % | Main drop reasons |
|--------|-------|------|--------|-------------------|
| gsm8k | 1,319 | 1,139 | 86.4% | 105 wrong answer, 59 too long |
| math | 200 | 107 | 53.5% | 55 wrong answer, 20 too long, 13 incoherent |
| arc | 400 | 346 | 86.5% | 44 no reasoning structure |
| humaneval | 164 | 152 | 92.7% | 11 too long |

### Kimi K2.5: 1,802/2,083 kept (86.5%)

| Source | Total | Kept | Keep % | Main drop reasons |
|--------|-------|------|--------|-------------------|
| gsm8k | 1,319 | 1,118 | 84.8% | 189 wrong answer |
| math | 200 | 128 | 64.0% | 46 wrong answer, 16 incoherent |
| arc | 400 | 395 | 98.8% | 3 too long |
| humaneval | 164 | 161 | 98.2% | 3 too long |

### Teacher Comparison

| Metric | GLM-5 | Kimi K2.5 |
|--------|-------|-----------|
| **Overall keep rate** | 83.7% | 86.5% |
| **Wrong answers dropped** | 160 (7.7%) | 235 (11.3%) |
| **Too long dropped** | 98 (4.7%) | 21 (1.0%) |
| **Median thinking tokens** | 433 | 325 |
| **Mean thinking tokens** | 676 | 538 |

**Key observations:**
1. **Kimi gets more answers wrong** (11.3% vs 7.7%) — GLM-5 is the more accurate reasoner, consistent with its AIME benchmark lead
2. **GLM-5 is way more verbose** — 98 traces exceeded 4,000 tokens vs Kimi's 21. GLM-5 over-explains.
3. **Kimi crushes ARC** (98.8% keep) — concise reasoning works really well for multiple choice
4. **MATH is hard for both** but Kimi keeps more (64% vs 53.5%) — surprising, possibly because shorter traces are less likely to go off the rails

---

## Phase 3: Formatting — Final Results

Ran `scripts/format_for_sft.py` with stratified splitting on both.

### GLM-5 Dataset

| Split | gsm8k | math | arc | humaneval | Total |
|-------|-------|------|-----|-----------|-------|
| Train | 911 | 85 | 276 | 121 | 1,393 |
| Validation | 113 | 10 | 34 | 15 | 172 |
| Test | 115 | 12 | 36 | 16 | 179 |

Token stats: median 824 total / 577 thinking, mean 1,139 total / 902 thinking.

### Kimi Dataset

| Split | gsm8k | math | arc | humaneval | Total |
|-------|-------|------|-----|-----------|-------|
| Train | 894 | 102 | 316 | 128 | 1,440 |
| Validation | 111 | 12 | 39 | 16 | 178 |
| Test | 113 | 14 | 40 | 17 | 184 |

Token stats: median 693 total / 433 thinking, mean 978 total / 717 thinking.

All splits stratified by source — every domain represented proportionally in train/val/test

---

## Phase 4: Dataset Upload

Pushed all 4 datasets to HuggingFace. Merged test split back into train since we're evaluating on external benchmarks (GSM8K, MATH, ARC) — the held-out test set was redundant. Kept validation for training loss monitoring.

| Dataset | Train | Val | Link |
|---------|-------|-----|------|
| GLM-5 raw | 1,744 | — | [glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) |
| GLM-5 SFT | 1,572 | 172 | [glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft) |
| Kimi raw | 1,802 | — | [kimi-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces) |
| Kimi SFT | 1,624 | 178 | [kimi-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/kimi-reasoning-traces-sft) |

---

## Phase 5: SFT Training

### Tinker Pipeline Verified

Got the Tinker training pipeline working using the cookbook's own functions:
- `conversation_to_datum()` — converts our messages format to Tinker Datum objects with proper token shifting and weight masking
- `compute_mean_nll()` — computes loss from logprobs and weights
- `TrainOnWhat.LAST_ASSISTANT_MESSAGE` — only trains on the assistant's reasoning turn (not system/user prompts)
- LoRA rank 32 on all attention + MLP layers

**Debugging the API:** Took a few tries to get the datum construction right. The cookbook wraps a lot of complexity:
1. First attempt: tried to access `model_input.tokens` — doesn't exist, need `to_ints()`
2. Second attempt: TensorData types from Tinker aren't numpy arrays — can't use `np.dot` directly
3. Third attempt: used the wrong import path for `compute_mean_nll`
4. Final: used the cookbook's built-in `conversation_to_datum` and `compute_mean_nll` — works correctly

**Lesson:** When a library has high-level helper functions, use them instead of reimplementing the low-level details. The cookbook exists for a reason.

**Capacity issue:** Hit "Tinker backend is running short on capacity" on the test run. GPU availability fluctuates — the request queues and executes when a slot opens. Normal for credit-based GPU platforms.

### Revised Plan: 3 SFT + 3 GRPO on 4B Only

Originally planned 6 SFT runs (2 students × 3 teachers). Discovered Qwen3.5-2B isn't supported on Tinker — only 4B and up. Rather than shoehorning a different small model in, simplified the plan: **3 SFT runs on Qwen3.5-4B, then GRPO on all 3.**

The 2B experiment can happen later on Colab Pro (Unsloth supports any HuggingFace model).

| Run | Student | Teacher | Dataset | Stage |
|-----|---------|---------|---------|-------|
| 1 | Qwen3.5-4B | GLM-5 | train_glm5.jsonl (1,572) | SFT |
| 2 | Qwen3.5-4B | Kimi | train_kimi.jsonl (1,624) | SFT |
| 3 | Qwen3.5-4B | Combined | train_combined.jsonl (3,196) | SFT |
| 4 | Qwen3.5-4B | GLM-5 | (RL on run 1) | GRPO |
| 5 | Qwen3.5-4B | Kimi | (RL on run 2) | GRPO |
| 6 | Qwen3.5-4B | Combined | (RL on run 3) | GRPO |

Config: LoRA rank 32, LR from `get_lr()`, AdamW (β1=0.9, β2=0.95), linear decay, batch 8.

**7 eval points:** 1 baseline + 3 after SFT + 3 after GRPO.

### Tinker Test Run — Success!

Capacity freed up and the test completed:

```
Step 1: loss=0.7519 (116.8s)
Step 2: loss=0.5947 (11.9s)
Step 3: loss=0.4150 (17.3s)
```

Loss dropping from 0.75 → 0.41 in 3 steps on 4 examples. The pipeline is correct. First step was slow (116s) due to model loading, subsequent steps fast (~12-17s).

*Ready to run all 6 SFT training runs.*

---

## Phase 6: Evaluation

### The Contamination Disaster

First eval run showed distilled models at 75-80% on GSM8K vs 35% base. Looked amazing. Too amazing.

Then I checked: **94% of our GSM8K eval problems were in the training data.** We trained on the GSM8K test split, then evaluated on... the GSM8K test split. The model wasn't reasoning — it was reciting memorized answers.

This is **benchmark contamination** — one of the most common mistakes in ML evaluation, and exactly the kind of thing that makes published results unreliable. We almost published garbage numbers.

**What went wrong:**
- `download_problems.py` pulled GSM8K **test** split (1,319 problems)
- Generated traces for all of them
- Trained on those traces
- `evaluate.py` sampled from GSM8K **test** split
- 94% overlap = model has seen these exact problems

**The fix — clean benchmarks with zero overlap:**

| Benchmark | Strategy | Contamination |
|-----------|----------|---------------|
| GSM8K **train split** | Training used test only — train is completely clean | 0% |
| MATH (seed 999) | Training sampled 200 with seed 42 — different seed gives different problems | Verified 0% |
| ARC (seed 999) | Same approach — different seed avoids the 400 we trained on | Verified 0% |
| MMLU-Pro | Never in our training pipeline at all | 0% |
| Trick questions | Hand-written, not from any dataset | 0% |

Every eval now runs a contamination check before scoring — programmatically verifies zero overlap between eval set and training data.

**Lesson for the article:** Always verify your eval data is clean. A 2-minute overlap check would have caught this immediately. We got lucky catching it before publishing.

### Eval Setup — 8 Models, 4 Benchmarks

| # | Model | Params | Why |
|---|-------|--------|-----|
| 1 | Base Qwen3.5-4B | 4B | Our baseline |
| 2 | **Distilled GLM-5** | 4B | Teacher comparison |
| 3 | **Distilled Kimi** | 4B | Teacher comparison |
| 4 | **Distilled Combined** | 4B | Best of both? |
| 5 | Llama-3.2-3B | 3B | Different architecture |
| 6 | Qwen3-8B | 8B | 2x our size, no distillation |
| 7 | gpt-oss-20b | 20B | OpenAI reference |
| 8 | Qwen3.5-27B | 27B | Upper bound |

All evaluated on: GSM8K (train split, 100), MATH (100), ARC (100), MMLU-Pro (100), 5 trick questions.

*Running all 8 in parallel on Tinker...*

### Billing Scare + Resume Logic

All 8 evals died simultaneously with `Error code: 402 — billing status blocked`. Thought we burned through $150 in credits. Turns out it was a temporary billing glitch — Tinker console still showed $100+ remaining, and sampling started working again a few minutes later.

But it exposed a critical flaw: the eval script had **no resume logic**. Every restart started from scratch, wasting all completed inference calls. Rewrote `eval_one.py` with incremental progress files:

- Each problem result saves to `data/eval_progress/{model}_{benchmark}.jsonl` immediately
- On restart, loads progress file and skips completed problems
- Also added 3x retry with backoff on transient Tinker errors

This is the same pattern as our trace generation script. Should have built it this way from the start.

### Early Clean Results + Emerging Hypothesis

With ~15-70 problems per model completed (clean, uncontaminated benchmarks), an interesting pattern:

| Teacher | Reasoning style | GSM8K (clean) | Median thinking tokens in training |
|---------|----------------|--------------|-----------------------------------|
| Kimi K2.5 | Concise | **71%** (17 problems) | 325 |
| Combined | Mixed | **67%** (15 problems) | ~430 |
| GLM-5 | Verbose | **53%** (15 problems) | 433 |
| Base (no distillation) | — | **25%** (16 problems) | 0 |

**Emerging hypothesis: concise teachers produce better small students.**

Why this might be happening:
1. **Capacity ceiling** — a 4B model has limited working memory. A 6,000 token GLM-5 reasoning chain overwhelms what the model can coherently reproduce. It learns to *start* reasoning but can't sustain it.
2. **Signal-to-noise** — GLM-5's verbose traces include restating, double-checking, and elaboration. A small model can't distinguish core reasoning from filler. Kimi's cleaner traces are higher signal.
3. **Training efficiency** — same gradient updates, but Kimi's shorter traces mean the model sees the full problem→answer arc more times per epoch. GLM-5's long traces get truncated at `max_length=4096`, sometimes cutting off the answer entirely.

This is counterintuitive — you'd expect richer, more detailed reasoning to help. But a 4B student can't absorb it all. **The teacher needs to match the student's capacity.**

Very preliminary (15-17 problems each). Need 100+ to confirm. But if it holds, this is the headline finding: "When distilling into small models, concise teachers beat verbose ones."

### Format Compliance: 0% But That's OK

Checked thinking tokens and `<think>`/`<answer>` format compliance — **0% across all models including distilled.** Panicked for a second. But looking at the actual output from our best model (distilled Kimi on the bat & ball problem):

> "Let me denote: Let b = cost of the ball... Substituting equation 2 into equation 1: b + (b + 1.00) = 1.10... 2b = 0.10... b = 0.05... Let me verify: Ball $0.05, Bat $1.05, Total $1.10 ✓"

The model IS reasoning step-by-step. It's just not wrapping it in `<think>`/`<answer>` tags. The Tinker/Qwen renderer uses the native chat template which doesn't trigger our custom tags.

**The distillation transferred the reasoning ability, not just the format.** The model learned to:
- Set up variables
- Write equations
- Solve step by step
- Verify the answer
- Flag common mistakes

These are the reasoning patterns from the teacher traces. The tags are cosmetic — the skill is real. For the article, this is actually a stronger finding: distillation teaches *how to think*, not just how to format output.

Reference models for context:
- Llama-3.2-3B (3B, no distillation): 10% — raw small base can't do math
- Qwen3-8B (8B, no distillation): 67% — our distilled 4B Kimi matches a model 2x its size
- gpt-oss-20b (20B): 84% — the ceiling
- Qwen3.5-27B (27B): 37% — base model without instruct tuning, can't follow instructions well

### Tinker Inference is Slow

Each eval call takes 3-9 seconds over the network (round-trip to Tinker GPU, generate tokens, send back). With 500 problems × 4 benchmarks × 6 models = 12,000 calls, that's ~20+ hours total. The models share Tinker's bandwidth so they slow each other down running in parallel.

**Decision:** Finish SFT eval on Tinker (it's running, progress saved incrementally). Do GRPO training on Tinker (training is fast, ~1hr per model). Then download the GRPO'd model weights and do ALL the post-GRPO re-evaluation on **Colab Pro** where inference is local GPU — 10-100x faster than API calls.

### Answer Extraction Bug

Discovered that base models (4B, 27B, 8B) were scoring lower than they should because our `extract_final_number()` just grabbed the last number in the response. With verbose reasoning that includes lots of intermediate numbers (step 1: 3+5=8, step 2: 8-2=6), the extractor sometimes grabbed the wrong number.

Fixed with smarter extraction that looks for explicit answer patterns first ("the answer is X", "therefore X", `\boxed{X}`, **X**) before falling back to last-number. Also bumped `max_tokens` from 1024 to 2048 so long responses don't get truncated before the final answer.

The 27B model at 37% is still suspiciously low. It's the base model (not instruct), so it generates reasoning but doesn't reliably format a clean final answer. This is a genuine limitation of base models, not an extraction bug.

---

### Colab Pro Notebook for Post-GRPO Eval & Publish

Wrote `notebooks/eval_and_publish.ipynb` for the final pipeline on Colab Pro:

1. **Download LoRA weights** from Tinker via `get_checkpoint_archive_url()`
2. **Merge with base model** using Unsloth `save_pretrained_merged()`
3. **Fast local eval** on A100 — same benchmarks (GSM8K train, MATH seed 999, ARC seed 999), ~10-100x faster than Tinker API
4. **Side-by-side trick questions** — all models on same 5 problems
5. **Push to HuggingFace** — merged 16-bit model
6. **GGUF export** — q4_k_m and q8_0 for local inference via Ollama

Tinker doesn't publish directly to HuggingFace — we download the LoRA adapters, merge locally, then push. The Colab notebook handles the full flow.

The checkpoint paths in the notebook point to SFT finals right now. After GRPO, we'll update them to the GRPO checkpoint paths and re-run.

### Benchmark Methodology Reality Check

Went looking for published benchmark numbers to validate our results. Found a big problem:

| Model | Published GSM8K | Our GSM8K (0-shot) | Gap |
|-------|----------------|-------------------|-----|
| gpt-oss-20b | 68.9% (published) | 84.6% (our eval) | +15.7 |
| Qwen3-4B-Base | 74.1% (5-shot) | 37.3% (our eval) | -36.8 |

**The gap is prompting methodology.** Published benchmarks use **5-shot prompting** for GSM8K (give the model 5 solved examples first). We're doing **zero-shot** (no examples). 5-shot massively helps base models but matters less for instruct-tuned or distilled models that already know the format.

**What this means:**
- Our internal comparisons (base → distilled) are valid — same methodology throughout
- But we can't directly compare our numbers to published model cards
- The +35 point distillation lift is real (both zero-shot)
- To compare against Jackrong's Claude-distilled Qwen3.5-4B, we need to run their exact methodology

**Fix:** Added both zero-shot AND 5-shot evaluation to the Colab notebook. Also added GPQA Diamond (Jackrong reports 38.9% on their distilled 4B — direct comparison). The Colab eval will run:

| Benchmark | Zero-shot (our method) | Few-shot (published method) |
|-----------|----------------------|---------------------------|
| GSM8K | ✅ | ✅ 5-shot |
| MATH | ✅ | — |
| ARC | ✅ | — |
| MMLU-Pro | ✅ | — |
| GPQA Diamond | — | ✅ 0-shot (Jackrong comparison) |
| Trick questions | ✅ | — |

**Lesson for the article:** Benchmark numbers without methodology context are meaningless. A model can score 37% or 74% on the same benchmark depending on whether you give it examples first. Always report your methodology.

### Eval Extraction Disaster

After burning thousands of Tinker inference calls, discovered that most non-GSM8K results were garbage:

| Issue | Affected | Root cause |
|-------|----------|-----------|
| MATH 0% on base models | base-4b, llama-3b, qwen35-27b | Extraction only checks `\boxed{}` — base models don't use that format |
| ARC 100% on base-4b | base-4b | Letter extraction too generous, matching random capitals in response |
| All MMLU-Pro untested | — | Never verified extraction worked before launching |

**What should have happened:** Test extraction on 5-10 problems per model × benchmark BEFORE launching 16K inference calls. Instead I tested the pipeline on one model (distilled Kimi on GSM8K), saw it work, and assumed it generalized. It didn't. Each model outputs answers in different formats.

**What's salvageable:**
- GSM8K for all models ✅ (number extraction works regardless of format)
- MATH for distilled models + gpt-oss-20b ✅ (they use `\boxed{}`)
- Everything else ❌

**Decision:** Stop Tinker evals. Move ALL benchmarking to Colab Pro where we can:
1. Test each model × benchmark extraction on 10 examples first
2. Run locally with fast GPU (no API latency)
3. Include both zero-shot and few-shot
4. Not burn credits on broken extraction

The GSM8K numbers are the headline anyway — distillation +35 points is real and solid. Full multi-benchmark comparison happens on Colab Pro post-GRPO.

**Lesson:** Never trust your eval pipeline until you've manually inspected outputs from EVERY model on EVERY benchmark. Automated extraction is fragile — different models format answers differently.

### The Right Way: lm-evaluation-harness

After the extraction disaster, researched how the industry actually does this. The answer: **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** by EleutherAI. It's the backend for the HuggingFace Open LLM Leaderboard. Every model card you've ever seen uses it.

Key insight: **ARC, GPQA, and MMLU-Pro don't use generation at all.** They use log-likelihood scoring — the model scores each multiple choice option and picks the highest probability. No answer extraction needed. Our custom regex approach was fundamentally wrong for these benchmarks.

Standard settings from Open LLM Leaderboard v2:

| Benchmark | Shots | Method | Task name |
|-----------|-------|--------|-----------|
| GSM8K | 8-shot CoT | Generative, regex | `gsm8k_cot` |
| MATH | 4-shot | Generative, `\boxed{}` | `minerva_math` |
| ARC-Challenge | 25-shot | Log-likelihood | `arc_challenge` |
| GPQA Diamond | 0-shot | Log-likelihood | `gpqa_diamond` |
| MMLU-Pro | 5-shot | Log-likelihood | `mmlu_pro` |

**Critical for MATH:** Must use `minerva_math` task (handles `\boxed{}`), NOT `leaderboard_math_hard` (which requires the model to output "The final answer is X. I hope it is correct." — a format our reasoning models don't use).

Rewrote the entire Colab eval notebook to use `lm-eval` instead of custom code. One `lm-eval` command per model, results directly comparable to every model card on HuggingFace. Also runs gpt-oss-20b locally on the H100 (80GB VRAM, fits in bf16).

**Lesson for the article:** Don't reinvent the wheel on eval. Use the standard tools. The time I spent writing custom extraction code was entirely wasted — `lm-eval` does it better, handles all the edge cases, and makes results comparable to published benchmarks.

---

## Phase 7: Export and Publish

### Checkpoint Format Disaster + Pivot to All-Colab

Tried to start GRPO on Tinker. Failed immediately: `Error code: 400 - Path is invalid`.

**Root cause:** During SFT training, we saved **sampler checkpoints** (`save_weights_for_sampler()`) which are inference-only LoRA weights. GRPO needs **state checkpoints** (`save_state()`) which include optimizer state for resuming training. We never saved those.

This was a planning failure. Should have mapped the full pipeline end-to-end before training:
- What does GRPO need as input? → state checkpoints
- What does our SFT script save? → sampler checkpoints
- Are those compatible? → NO

**The fix:** Move everything to Colab Pro. The sampler checkpoints ARE downloadable LoRA weights — we can load them with Unsloth's `from_pretrained()` and run GRPO locally using TRL's `GRPOTrainer`. No Tinker credits needed for any remaining step.

Downloaded all 3 LoRA adapters (278MB each):
- `sft_lora_glm5/adapter_model.safetensors` + `adapter_config.json`
- `sft_lora_kimi/adapter_model.safetensors` + `adapter_config.json`
- `sft_lora_combined/adapter_model.safetensors` + `adapter_config.json`

**New pipeline (everything on Colab Pro H100):**

| Step | What | How |
|------|------|-----|
| 1. Load SFT LoRA | Download from Tinker or upload from local | Free file transfer |
| 2. GRPO training | TRL `GRPOTrainer` with GSM8K rewards | Local H100 GPU |
| 3. Benchmark | `lm-evaluation-harness` on all models | Local H100 GPU |
| 4. Merge + publish | Unsloth merge → HuggingFace → GGUF | Local H100 GPU |

**Zero additional Tinker credits.** Everything from here runs on the H100.

Rewrote the Colab notebook from scratch with this full pipeline. Removed the Tinker GRPO script.

**Lessons:**
1. Map the FULL pipeline end-to-end before spending money on any step
2. Verify checkpoint compatibility before training
3. `save_weights_for_sampler` ≠ `save_state` — one is for inference, one is for resuming training
4. When in doubt, save BOTH types

*Moving to Colab Pro for GRPO + eval + publish...*

### Colab Notebook Hell

What should have been "run the notebook" turned into hours of debugging:

1. **`pip install unsloth` pins transformers<=4.57.6** — Qwen3.5 needs transformers 5.3.0. Fix: use Unsloth's official install script (`curl -fsSL https://unsloth.ai/install.sh | sh`).

2. **Unsloth install script installs transformers 5.0.0** — still too old for qwen3_5. Fix: explicitly `pip install transformers==5.3.0` after.

3. **`fast_inference=True` requires vllm** — which wasn't installed. Fix: remove fast_inference (not needed for training).

4. **`get_peft_model()` fails with "already added LoRA"** — the Tinker checkpoint already has LoRA adapters attached. Fix: remove get_peft_model call.

5. **`TypeError: string indices must be integers, not 'str'`** — Qwen3.5-4B is architecturally a VLM (`Qwen3_5ForConditionalGeneration`). `from_pretrained` returns a Processor (wraps tokenizer + image processor), not a plain Tokenizer. GRPOTrainer passes the Processor to `apply_chat_template`, which tries to iterate message content looking for images — fails on plain strings. Fix: extract the plain tokenizer with `tokenizer = tokenizer.tokenizer`.

**This last one was verified locally** by reproducing the exact error without GPU:
```python
proc = AutoProcessor.from_pretrained('Qwen/Qwen3.5-4B')
for message in msgs:
    visuals = [content for content in message["content"] if content["type"] in ["image", "video"]]
# TypeError: string indices must be integers, not 'str'
```

**Lessons:**
- Qwen3.5 is natively multimodal. There is no text-only variant. Every Qwen3.5 model has vision components even if you only use text.
- Always test locally what you can before burning GPU time. The tokenizer/processor behavior can be verified without a GPU.
- Read the actual error traceback carefully. The error was in `transformers/processing_utils.py`, not in our code — pointing to the Processor vs Tokenizer issue.
