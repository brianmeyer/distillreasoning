"""Post-generation pipeline: filter → format → upload → train.

Runs all steps sequentially and writes rich devlog entries after each step.
Called by Claude when trace generation is confirmed complete.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

DEVLOG = Path("DEVLOG.md")
FILTER_REPORT = Path("data/filter_report.json")
FORMAT_REPORT = Path("data/format_report.json")


def run(cmd, env=None):
    """Run a command, stream output, raise on failure."""
    print(f"\n$ {' '.join(cmd)}")
    full_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, env=full_env)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{result.stderr}")
    return result.stdout


def append_devlog(text):
    with open(DEVLOG, "a") as f:
        f.write(text)
    print(f"[devlog] Written {len(text)} chars")


def step_filter():
    print("\n" + "="*60)
    print("STEP 1: Filtering traces")
    print("="*60)
    run(["./venv/bin/python", "scripts/filter_traces.py"])

    report = json.loads(FILTER_REPORT.read_text())
    s = report["summary"]
    by_source = report["by_source"]
    tl = report["thinking_length_distribution"]

    # Build devlog entry
    kept_ex = report.get("kept_examples", {})
    drop_ex = report.get("drop_examples", {})

    # Format source table
    source_rows = ""
    for src, d in by_source.items():
        if d["total"] > 0:
            reasons = ", ".join(f"{v} {k}" for k, v in d.get("drop_reasons", {}).items())
            source_rows += f"| {src:12s} | {d['total']:5d} | {d['kept']:5d} | {d['keep_rate_pct']:5.1f}% | {reasons or '—'} |\n"

    # Format kept examples
    kept_section = ""
    for src, examples in kept_ex.items():
        for ex in examples[:1]:
            kept_section += f"""
**Kept example ({src})** — {ex['thinking_tokens']} thinking tokens
> Problem: {ex['problem'][:200]}

> Thinking (first 300 chars): {ex['thinking_preview'][:300]}

> Response: {ex['response_preview'][:200]}
"""

    # Format dropped examples
    drop_section = ""
    for reason, examples in drop_ex.items():
        if examples:
            ex = examples[0]
            drop_section += f"""
**Dropped ({reason})**
> Problem: {ex['problem'][:200]}
> Reason: {reason}
"""
            if "thinking_tokens" in ex:
                drop_section += f"> Thinking tokens: {ex['thinking_tokens']} (minimum is 50)\n"
            if "expected" in ex:
                drop_section += f"> Expected: {ex['expected']} | Model tail: ...{ex.get('model_response_tail','')[-100:]}\n"

    entry = f"""
---

## Phase 2: Filtering — {datetime.now().strftime('%B %d, %Y %I:%M %p')}

Trace generation finished. Ran `scripts/filter_traces.py` to drop low-quality traces.

### Filter results

| Source | Total | Kept | Keep rate | Drop reasons |
|--------|-------|------|-----------|-------------|
{source_rows}
**Overall: {s['total_kept']}/{s['total_input']} kept ({s['keep_rate_pct']}%)**

### Thinking length distribution (tokens)

| Stat | Value |
|------|-------|
| Min | {tl.get('min', '?')} |
| 25th percentile | {tl.get('p25', '?')} |
| Median | {tl.get('median', '?')} |
| 75th percentile | {tl.get('p75', '?')} |
| Max | {tl.get('max', '?')} |
| Mean | {tl.get('mean', '?')} |

### Examples of kept traces
{kept_section}

### Examples of dropped traces
{drop_section}
"""
    append_devlog(entry)
    return report


def step_format():
    print("\n" + "="*60)
    print("STEP 2: Formatting for SFT")
    print("="*60)
    run(["./venv/bin/python", "scripts/format_for_sft.py"])

    report = json.loads(FORMAT_REPORT.read_text())
    s = report["summary"]
    ts = report["token_stats"]
    samples = report.get("samples", [])

    sample_section = ""
    for i, sample in enumerate(samples[:1]):
        sample_section += f"""
**Sample formatted entry ({sample['source']})**

Problem:
```
{sample['problem'][:300]}
```

Expected answer: `{sample['expected_answer']}`

Formatted assistant turn ({sample['total_chars']} chars):
```
{sample['formatted_assistant_preview'][:600]}
...
```
"""

    entry = f"""
---

## Phase 3: Formatting — {datetime.now().strftime('%B %d, %Y %I:%M %p')}

Ran `scripts/format_for_sft.py` to convert filtered traces into SFT chat format.

### Split

| Split | Count | Purpose |
|-------|-------|---------|
| Train | {s['train']} | Model learns from these |
| Validation | {s['validation']} | Monitor loss during training |
| Test | {s['test']} | Final held-out evaluation |
| **Total** | **{s['total']}** | 80/10/10 split |

### Token length statistics

| | Full assistant turn | Thinking only | Answer only |
|--|--------------------|--------------------|-------------|
| Min | {ts['full_assistant_turn']['min']} | {ts['thinking_section']['min']} | {ts['answer_section']['min']} |
| Median | {ts['full_assistant_turn']['median']} | {ts['thinking_section']['median']} | {ts['answer_section']['median']} |
| Mean | {ts['full_assistant_turn']['mean']} | {ts['thinking_section']['mean']} | {ts['answer_section']['mean']} |
| Max | {ts['full_assistant_turn']['max']} | {ts['thinking_section']['max']} | {ts['answer_section']['max']} |

### Format

Every training example follows this template:

```
[system] You are a helpful reasoning assistant. Think through problems step by step before answering.
[user]   <the problem>
[assistant] <think>
             <GLM-5 full reasoning chain>
             </think>

             <answer>
             <final response>
             </answer>
```

The `<think>`/`<answer>` tags teach the model to produce *structured* reasoning — not just a flat answer. During training, only the assistant turn contributes to the loss (`train_on_responses_only`).
{sample_section}
"""
    append_devlog(entry)
    return report


def step_upload(hf_token):
    print("\n" + "="*60)
    print("STEP 3: Uploading to HuggingFace")
    print("="*60)
    run(["./venv/bin/python", "scripts/upload_dataset.py"], env={"HF_TOKEN": hf_token})

    entry = f"""
---

## Phase 4: Dataset Upload — {datetime.now().strftime('%B %d, %Y %I:%M %p')}

Both datasets pushed to HuggingFace:

- **Raw traces:** https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces
- **SFT-formatted:** https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft
"""
    append_devlog(entry)


def step_train(tinker_key):
    print("\n" + "="*60)
    print("STEP 4: SFT Training on Tinker")
    print("="*60)

    entry = f"""
---

## Phase 5: SFT Training — {datetime.now().strftime('%B %d, %Y %I:%M %p')}

Kicked off training on Tinker (Thinking Machines Lab).

**Config:**
- Model: Qwen/Qwen3.5-4B
- Method: LoRA rank 32 (all attention + MLP layers)
- Epochs: 3
- Batch size: 128
- LR: recommended by `get_lr("Qwen/Qwen3.5-4B")`
- Loss: cross-entropy on assistant turns only

Training in progress — will update with loss curves and results when complete.
"""
    append_devlog(entry)

    # Run training (this will block for ~1-2 hrs)
    run(["./venv/bin/python", "scripts/train_tinker.py"], env={"TINKER_API_KEY": tinker_key})


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN", "")
    tinker_key = os.environ.get("TINKER_API_KEY", "")

    if not hf_token:
        print("Set HF_TOKEN env var")
        sys.exit(1)
    if not tinker_key:
        print("Set TINKER_API_KEY env var")
        sys.exit(1)

    step_filter()
    step_format()
    step_upload(hf_token)
    step_train(tinker_key)
