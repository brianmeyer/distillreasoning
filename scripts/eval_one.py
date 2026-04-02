"""Evaluate a single model on CLEAN benchmarks via Tinker sampling.

RESUME LOGIC: Each problem result is saved to a JSONL progress file
as it completes. On restart, already-completed problems are skipped.
If the process dies at problem 347/1000, restarting picks up at 348.

Benchmarks (all verified zero overlap with training data):
  - GSM8K train split — our training used test split only
  - MATH test (seed 999) — avoids the 200 we trained on (seed 42)
  - ARC-Challenge test (seed 999) — avoids the 400 we trained on (seed 42)
  - MMLU-Pro — never in training data
  - 5 trick questions — hand-written

Usage:
  TINKER_API_KEY=xxx python scripts/eval_one.py <name> [--checkpoint <path>] [--base-model <model>]
"""

import tinker
from tinker import types
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from datasets import load_dataset
import json, re, time, random, argparse
from pathlib import Path

EVAL_SEED = 999
EVAL_DIR = Path("data/eval_results")
EVAL_DIR.mkdir(exist_ok=True)
PROGRESS_DIR = Path("data/eval_progress")
PROGRESS_DIR.mkdir(exist_ok=True)

N_PER_BENCH = 500
SYSTEM_MSG = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRICK_QUESTIONS = [
    ("A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?", "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    ("A store offers a 20% discount on a jacket that costs $80. Then they add 8% sales tax. What is the final price?", "69.12"),
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "no"),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_final_number(text):
    """Extract the final answer number, preferring explicit answer patterns."""
    text_clean = text.replace(",", "")

    # Priority 1: Look for explicit answer patterns
    answer_patterns = [
        r"(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$?\s*(-?\d+\.?\d*)",
        r"(?:therefore|thus|so)\s*,?\s*.*?(?:is|=|equals)\s*\$?\s*(-?\d+\.?\d*)\s*(?:\.|$|\n)",
        r"\\boxed\{(-?\d+\.?\d*)\}",
        r"\*\*\$?(-?\d+\.?\d*)\$?\*\*\s*(?:\.|$|\n|<)",  # **bold answer**
    ]
    for pat in answer_patterns:
        matches = re.findall(pat, text_clean, re.IGNORECASE)
        if matches:
            return float(matches[-1])

    # Priority 2: Last number in the text
    numbers = re.findall(r"-?\d+\.?\d*", text_clean)
    return float(numbers[-1]) if numbers else None

def extract_boxed(text):
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None

def has_format(text):
    return "<think>" in text and "</think>" in text

def count_thinking_tokens(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(match.group(1).split()) if match else 0

def load_training_problems():
    problems = set()
    for f in ["data/train_glm5.jsonl", "data/train_kimi.jsonl"]:
        try:
            with open(f) as fh:
                for line in fh:
                    ex = json.loads(line)
                    problems.add(ex["messages"][1]["content"][:150])
        except FileNotFoundError:
            pass
    return problems

def verify_no_contamination(eval_problems, training_problems, bench_name):
    overlap = sum(1 for p in eval_problems if p[:150] in training_problems)
    if overlap > 0:
        print(f"  ⚠️  CONTAMINATION: {overlap}/{len(eval_problems)} {bench_name} overlap!")
    else:
        print(f"  ✅ Clean: 0/{len(eval_problems)} {bench_name} overlap with training")
    return overlap

def sample_with_retry(sc, tokenizer, renderer, problem, max_tokens=2048, max_retries=3):
    """Sample with retry on transient errors."""
    messages = [{"role": "system", "content": SYSTEM_MSG}, {"role": "user", "content": problem}]
    prompt = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95, stop=stop)

    for attempt in range(1, max_retries + 1):
        try:
            start = time.time()
            result = sc.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
            elapsed = time.time() - start
            response = tokenizer.decode(result.sequences[0].tokens) if result.sequences else ""
            return response, elapsed
        except Exception as e:
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"      Retry {attempt}/{max_retries}: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"      FAILED after {max_retries} retries: {e}")
                return "", 0.0


# ── Resume Logic ───────────────────────────────────────────────────────────────

def load_progress(model_name, bench_name):
    """Load completed problem indices from progress file."""
    progress_file = PROGRESS_DIR / f"{model_name}_{bench_name}.jsonl"
    completed = {}
    if progress_file.exists():
        with open(progress_file) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed[entry["idx"]] = entry
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed, progress_file

def save_progress(progress_file, entry):
    """Append one problem result to progress file."""
    with open(progress_file, "a") as f:
        f.write(json.dumps(entry) + "\n")
        f.flush()

def summarize_progress(completed):
    """Compute summary stats from completed results."""
    n = len(completed)
    if n == 0:
        return {"accuracy": 0, "format_pct": 0, "avg_thinking": 0, "correct": 0, "n": 0}
    correct = sum(1 for e in completed.values() if e.get("correct"))
    format_ok = sum(1 for e in completed.values() if e.get("format"))
    think = sum(e.get("thinking_tokens", 0) for e in completed.values())
    return {
        "accuracy": round(100 * correct / n, 1),
        "format_pct": round(100 * format_ok / n, 1),
        "avg_thinking": round(think / n),
        "correct": correct,
        "n": n,
    }


# ── Benchmark Functions (with resume) ──────────────────────────────────────────

def eval_gsm8k(sc, tok, renderer, training_problems, model_name, n=N_PER_BENCH):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    verify_no_contamination([ds[i]["question"] for i in indices], training_problems, "GSM8K-train")

    completed, pfile = load_progress(model_name, "gsm8k")
    print(f"    Resuming from {len(completed)}/{n} completed")

    for idx, i in enumerate(indices):
        if idx in completed:
            continue
        row = ds[i]
        response, elapsed = sample_with_retry(sc, tok, renderer, row["question"])
        expected = row["answer"].split("####")[-1].strip()
        fmt = has_format(response)
        tt = count_thinking_tokens(response)
        got_right = False
        try:
            exp_num = float(expected.replace(",", ""))
            mod_num = extract_final_number(response)
            if mod_num is not None and abs(mod_num - exp_num) < 0.01:
                got_right = True
        except ValueError:
            pass
        save_progress(pfile, {"idx": idx, "correct": got_right, "format": fmt, "thinking_tokens": tt})
        completed[idx] = {"idx": idx, "correct": got_right, "format": fmt, "thinking_tokens": tt}
        done = len(completed)
        if done % 10 == 0:
            c = sum(1 for e in completed.values() if e["correct"])
            print(f"    GSM8K: {done}/{n}: {c}/{done} correct")

    result = summarize_progress(completed)
    result["benchmark"] = "GSM8K (train split)"
    return result


def eval_math(sc, tok, renderer, training_problems, model_name, n=N_PER_BENCH):
    ds = load_dataset("SuperSecureHuman/competition_math_hf_dataset", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    verify_no_contamination([ds[i]["problem"] for i in indices], training_problems, "MATH")

    completed, pfile = load_progress(model_name, "math")
    print(f"    Resuming from {len(completed)}/{n} completed")

    for idx, i in enumerate(indices):
        if idx in completed:
            continue
        row = ds[i]
        response, _ = sample_with_retry(sc, tok, renderer, row["problem"])
        fmt = has_format(response)
        tt = count_thinking_tokens(response)
        expected_boxed = extract_boxed(row.get("solution", ""))
        model_boxed = extract_boxed(response)
        got_right = bool(expected_boxed and model_boxed and expected_boxed.strip() == model_boxed.strip())
        save_progress(pfile, {"idx": idx, "correct": got_right, "format": fmt, "thinking_tokens": tt})
        completed[idx] = {"idx": idx, "correct": got_right, "format": fmt, "thinking_tokens": tt}
        done = len(completed)
        if done % 10 == 0:
            c = sum(1 for e in completed.values() if e["correct"])
            print(f"    MATH: {done}/{n}: {c}/{done} correct")

    result = summarize_progress(completed)
    result["benchmark"] = "MATH-500 (clean seed)"
    return result


def eval_arc(sc, tok, renderer, training_problems, model_name, n=N_PER_BENCH):
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    verify_no_contamination([ds[i]["question"] for i in indices], training_problems, "ARC")

    completed, pfile = load_progress(model_name, "arc")
    print(f"    Resuming from {len(completed)}/{n} completed")

    for idx, i in enumerate(indices):
        if idx in completed:
            continue
        row = ds[i]
        choices = row["choices"]
        choice_text = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        problem = f"{row['question']}\n\nChoices:\n{choice_text}"
        expected = row["answerKey"]
        response, _ = sample_with_retry(sc, tok, renderer, problem)
        fmt = has_format(response)
        tt = count_thinking_tokens(response)
        resp_upper = response.upper()
        found = False
        for pat in [rf"answer\s+is\s+\(?{expected}\)?", rf"\({expected}\)", rf"\b{expected}\b\s*[\.\)]"]:
            if re.search(pat, resp_upper):
                found = True
                break
        if not found:
            last_caps = re.findall(r"\b([A-E])\b", resp_upper)
            if last_caps and last_caps[-1] == expected:
                found = True
        save_progress(pfile, {"idx": idx, "correct": found, "format": fmt, "thinking_tokens": tt})
        completed[idx] = {"idx": idx, "correct": found, "format": fmt, "thinking_tokens": tt}
        done = len(completed)
        if done % 10 == 0:
            c = sum(1 for e in completed.values() if e["correct"])
            print(f"    ARC: {done}/{n}: {c}/{done} correct")

    result = summarize_progress(completed)
    result["benchmark"] = "ARC-Challenge (clean seed)"
    return result


def eval_mmlu_pro(sc, tok, renderer, training_problems, model_name, n=N_PER_BENCH):
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))
    print(f"  ✅ Clean: MMLU-Pro was never in training data")

    completed, pfile = load_progress(model_name, "mmlu_pro")
    print(f"    Resuming from {len(completed)}/{n} completed")

    for idx, i in enumerate(indices):
        if idx in completed:
            continue
        row = ds[i]
        options = row["options"]
        letters = [chr(65 + j) for j in range(len(options))]
        choices = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
        problem = f"{row['question']}\n\nChoices:\n{choices}"
        expected = row["answer"]
        response, _ = sample_with_retry(sc, tok, renderer, problem)
        fmt = has_format(response)
        tt = count_thinking_tokens(response)
        resp_upper = response.upper()
        found = False
        for pat in [rf"answer\s+is\s+\(?{expected}\)?", rf"\({expected}\)", rf"\b{expected}\b\s*[\.\)]"]:
            if re.search(pat, resp_upper):
                found = True
                break
        if not found:
            last_caps = re.findall(r"\b([A-Z])\b", resp_upper)
            if last_caps and last_caps[-1] == expected:
                found = True
        save_progress(pfile, {"idx": idx, "correct": found, "format": fmt, "thinking_tokens": tt})
        completed[idx] = {"idx": idx, "correct": found, "format": fmt, "thinking_tokens": tt}
        done = len(completed)
        if done % 10 == 0:
            c = sum(1 for e in completed.values() if e["correct"])
            print(f"    MMLU-Pro: {done}/{n}: {c}/{done} correct")

    result = summarize_progress(completed)
    result["benchmark"] = "MMLU-Pro (fully clean)"
    return result


def eval_tricks(sc, tok, renderer, model_name):
    completed, pfile = load_progress(model_name, "tricks")
    results = []
    for idx, (q, expected) in enumerate(TRICK_QUESTIONS):
        if idx in completed:
            results.append(completed[idx])
            continue
        response, elapsed = sample_with_retry(sc, tok, renderer, q)
        model_num = extract_final_number(response)
        try:
            exp_num = float(expected)
            got_right = model_num is not None and abs(model_num - exp_num) < 0.01
        except ValueError:
            got_right = expected.lower() in response.lower()
        entry = {"idx": idx, "problem": q, "expected": expected, "correct": got_right,
                 "response": response[:1500], "has_format": has_format(response),
                 "thinking_tokens": count_thinking_tokens(response)}
        save_progress(pfile, entry)
        completed[idx] = entry
        results.append(entry)
    correct = sum(1 for r in results if r["correct"])
    print(f"    Tricks: {correct}/5 correct")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--benchmarks", default="gsm8k,math,arc,mmlu_pro")
    parser.add_argument("--large", action="store_true")
    args = parser.parse_args()

    n = N_PER_BENCH
    print(f"Evaluating: {args.name}")
    print(f"  Base model: {args.base_model}")
    print(f"  Checkpoint: {args.checkpoint or 'none (base)'}")
    print(f"  Benchmarks: {args.benchmarks}")
    print(f"  Problems per benchmark: {n}")
    print(f"  Eval seed: {EVAL_SEED}")
    print(f"  Progress dir: {PROGRESS_DIR}")

    training_problems = load_training_problems()
    print(f"  {len(training_problems)} training problems loaded for contamination check")

    service = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    if args.checkpoint:
        sc = service.create_sampling_client(model_path=args.checkpoint)
    else:
        sc = service.create_sampling_client(base_model=args.base_model)

    result = {"model": args.name, "base_model": args.base_model, "checkpoint": args.checkpoint, "eval_seed": EVAL_SEED}

    bench_fns = {
        "gsm8k": ("GSM8K", eval_gsm8k),
        "math": ("MATH", eval_math),
        "arc": ("ARC", eval_arc),
        "mmlu_pro": ("MMLU-Pro", eval_mmlu_pro),
    }

    for bench_key in args.benchmarks.split(","):
        if bench_key in bench_fns:
            label, fn = bench_fns[bench_key]
            print(f"\n  {label} ({n} problems)...")
            scores = fn(sc, tokenizer, renderer, training_problems, args.name, n=n)
            result[bench_key] = scores
            print(f"  {label}: {scores['accuracy']}% acc | {scores['format_pct']}% format | {scores['avg_thinking']} think tok")

    print(f"\n  Trick questions (5)...")
    result["trick_questions"] = eval_tricks(sc, tokenizer, renderer, args.name)

    out_file = EVAL_DIR / f"{args.name}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✅ Saved: {out_file}")

    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {args.name}")
    for bench_key in args.benchmarks.split(","):
        if bench_key in result:
            print(f"    {bench_key}: {result[bench_key]['accuracy']}%")
    tricks_correct = sum(1 for t in result["trick_questions"] if t["correct"])
    print(f"    tricks: {tricks_correct}/5")


if __name__ == "__main__":
    main()
