"""Evaluate a single model on CLEAN benchmarks via Tinker sampling.

All benchmarks verified to have ZERO overlap with training data.

Benchmarks:
  Tier 1 (completely clean):
    - GSM8K train split (100) — our training used test split only
    - MMLU-Pro (100) — never in training data
  Tier 2 (filtered clean):
    - MATH test (100, seed 999) — avoids the 200 we trained on (seed 42)
    - ARC-Challenge test (100, seed 999) — avoids the 400 we trained on (seed 42)
  Tier 3 (qualitative):
    - 5 trick questions — hand-written, not from any dataset

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

EVAL_SEED = 999  # Different from training seed (42) to avoid overlap
random.seed(EVAL_SEED)

EVAL_DIR = Path("data/eval_results")
EVAL_DIR.mkdir(exist_ok=True)

N_PER_BENCH = 500          # Default for reference models
N_PER_BENCH_DISTILLED = 1000  # More for our distilled + base (tighter confidence intervals)
SYSTEM_MSG = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRICK_QUESTIONS = [
    ("A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?", "9"),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?", "5"),
    ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?", "0.05"),
    ("A store offers a 20% discount on a jacket that costs $80. Then they add 8% sales tax. What is the final price?", "69.12"),
    ("If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?", "no"),
]


def extract_final_number(text):
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
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
    """Load all training problems for contamination checking."""
    problems = set()
    for f in ["data/train_glm5.jsonl", "data/train_kimi.jsonl"]:
        try:
            with open(f) as fh:
                for line in fh:
                    ex = json.loads(line)
                    # Use first 150 chars of user message as fingerprint
                    user_msg = ex["messages"][1]["content"][:150]
                    problems.add(user_msg)
        except FileNotFoundError:
            pass
    return problems


def verify_no_contamination(eval_problems, training_problems, bench_name):
    """Verify zero overlap between eval and training problems."""
    overlap = 0
    for p in eval_problems:
        if p[:150] in training_problems:
            overlap += 1
    if overlap > 0:
        print(f"  ⚠️  CONTAMINATION WARNING: {overlap}/{len(eval_problems)} {bench_name} eval problems found in training data!")
    else:
        print(f"  ✅ Clean: 0/{len(eval_problems)} {bench_name} eval problems overlap with training")
    return overlap


def sample(sc, tokenizer, renderer, problem, max_tokens=1024):
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": problem},
    ]
    prompt = renderer.build_generation_prompt(messages)
    stop = renderer.get_stop_sequences()
    params = types.SamplingParams(max_tokens=max_tokens, temperature=0.6, top_p=0.95, stop=stop)
    start = time.time()
    result = sc.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    elapsed = time.time() - start
    response = tokenizer.decode(result.sequences[0].tokens) if result.sequences else ""
    return response, elapsed


# ── Benchmark Functions ────────────────────────────────────────────────────────

def eval_gsm8k(sc, tok, renderer, training_problems, n=N_PER_BENCH):
    """GSM8K from TRAIN split (our training used TEST split — zero contamination)."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    # Verify clean
    eval_texts = [ds[i]["question"] for i in indices]
    verify_no_contamination(eval_texts, training_problems, "GSM8K-train")

    correct, format_ok, think_tokens = 0, 0, []
    for idx, i in enumerate(indices):
        row = ds[i]
        response, _ = sample(sc, tok, renderer, row["question"])
        expected = row["answer"].split("####")[-1].strip()
        if has_format(response): format_ok += 1
        think_tokens.append(count_thinking_tokens(response))
        try:
            exp_num = float(expected.replace(",", ""))
            mod_num = extract_final_number(response)
            if mod_num is not None and abs(mod_num - exp_num) < 0.01:
                correct += 1
        except ValueError:
            pass
        if (idx + 1) % 10 == 0:
            print(f"    GSM8K: {idx+1}/{n}: {correct}/{idx+1} correct")

    return {"benchmark": "GSM8K (train split)", "accuracy": round(100 * correct / n, 1),
            "format_pct": round(100 * format_ok / n, 1),
            "avg_thinking": round(sum(think_tokens) / n), "correct": correct, "n": n}


def eval_math(sc, tok, renderer, training_problems, n=N_PER_BENCH):
    """MATH test with seed 999 (training used seed 42 — different 100 problems)."""
    ds = load_dataset("SuperSecureHuman/competition_math_hf_dataset", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    eval_texts = [ds[i]["problem"] for i in indices]
    verify_no_contamination(eval_texts, training_problems, "MATH")

    correct, format_ok, think_tokens = 0, 0, []
    for idx, i in enumerate(indices):
        row = ds[i]
        response, _ = sample(sc, tok, renderer, row["problem"])
        expected_boxed = extract_boxed(row.get("solution", ""))
        if has_format(response): format_ok += 1
        think_tokens.append(count_thinking_tokens(response))
        model_boxed = extract_boxed(response)
        if expected_boxed and model_boxed and expected_boxed.strip() == model_boxed.strip():
            correct += 1
        if (idx + 1) % 10 == 0:
            print(f"    MATH: {idx+1}/{n}: {correct}/{idx+1} correct")

    return {"benchmark": "MATH-500 (clean seed)", "accuracy": round(100 * correct / n, 1),
            "format_pct": round(100 * format_ok / n, 1),
            "avg_thinking": round(sum(think_tokens) / n), "correct": correct, "n": n}


def eval_arc(sc, tok, renderer, training_problems, n=N_PER_BENCH):
    """ARC-Challenge test with seed 999 (training used seed 42)."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    eval_texts = [ds[i]["question"] for i in indices]
    verify_no_contamination(eval_texts, training_problems, "ARC")

    correct, format_ok, think_tokens = 0, 0, []
    for idx, i in enumerate(indices):
        row = ds[i]
        choices = row["choices"]
        choice_text = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        problem = f"{row['question']}\n\nChoices:\n{choice_text}"
        expected = row["answerKey"]

        response, _ = sample(sc, tok, renderer, problem)
        if has_format(response): format_ok += 1
        think_tokens.append(count_thinking_tokens(response))

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
        if found:
            correct += 1
        if (idx + 1) % 10 == 0:
            print(f"    ARC: {idx+1}/{n}: {correct}/{idx+1} correct")

    return {"benchmark": "ARC-Challenge (clean seed)", "accuracy": round(100 * correct / n, 1),
            "format_pct": round(100 * format_ok / n, 1),
            "avg_thinking": round(sum(think_tokens) / n), "correct": correct, "n": n}


def eval_mmlu_pro(sc, tok, renderer, training_problems, n=N_PER_BENCH):
    """MMLU-Pro — completely clean, never in training data."""
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    rng = random.Random(EVAL_SEED)
    indices = rng.sample(range(len(ds)), min(n, len(ds)))

    correct, format_ok, think_tokens = 0, 0, []
    print(f"  ✅ Clean: MMLU-Pro was never in training data")

    for idx, i in enumerate(indices):
        row = ds[i]
        options = row["options"]
        letters = [chr(65 + j) for j in range(len(options))]
        choices = "\n".join(f"{l}. {o}" for l, o in zip(letters, options))
        problem = f"{row['question']}\n\nChoices:\n{choices}"
        expected = row["answer"]

        response, _ = sample(sc, tok, renderer, problem)
        if has_format(response): format_ok += 1
        think_tokens.append(count_thinking_tokens(response))

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
        if found:
            correct += 1
        if (idx + 1) % 10 == 0:
            print(f"    MMLU-Pro: {idx+1}/{n}: {correct}/{idx+1} correct")

    return {"benchmark": "MMLU-Pro (fully clean)", "accuracy": round(100 * correct / n, 1),
            "format_pct": round(100 * format_ok / n, 1),
            "avg_thinking": round(sum(think_tokens) / n), "correct": correct, "n": n}


def eval_tricks(sc, tok, renderer):
    """Hand-written trick questions — not from any dataset."""
    results = []
    for q, expected in TRICK_QUESTIONS:
        response, elapsed = sample(sc, tok, renderer, q)
        model_num = extract_final_number(response)
        try:
            exp_num = float(expected)
            got_right = model_num is not None and abs(model_num - exp_num) < 0.01
        except ValueError:
            got_right = expected.lower() in response.lower()

        results.append({
            "problem": q, "expected": expected, "correct": got_right,
            "response": response[:1500], "has_format": has_format(response),
            "thinking_tokens": count_thinking_tokens(response),
        })
    correct = sum(1 for r in results if r["correct"])
    print(f"    Tricks: {correct}/5 correct")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--benchmarks", default="gsm8k,math,arc,mmlu_pro",
                        help="Comma-separated: gsm8k,math,arc,mmlu_pro")
    parser.add_argument("--large", action="store_true",
                        help="Use 1000 problems per benchmark (for distilled + base)")
    args = parser.parse_args()

    print(f"Evaluating: {args.name}")
    print(f"  Base model: {args.base_model}")
    print(f"  Checkpoint: {args.checkpoint or 'none (base)'}")
    print(f"  Benchmarks: {args.benchmarks}")
    n = N_PER_BENCH_DISTILLED if args.large else N_PER_BENCH
    print(f"  Eval seed: {EVAL_SEED} (training used 42)")
    print(f"  Problems per benchmark: {n}")

    # Load training data for contamination check
    print(f"\n  Loading training data for contamination verification...")
    training_problems = load_training_problems()
    print(f"  {len(training_problems)} unique training problems loaded")

    service = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(args.base_model)
    renderer_name = model_info.get_recommended_renderer_name(args.base_model)
    renderer = get_renderer(renderer_name, tokenizer)

    if args.checkpoint:
        sc = service.create_sampling_client(model_path=args.checkpoint)
    else:
        sc = service.create_sampling_client(base_model=args.base_model)

    result = {"model": args.name, "base_model": args.base_model,
              "checkpoint": args.checkpoint, "eval_seed": EVAL_SEED}

    bench_fns = {
        "gsm8k": ("GSM8K", eval_gsm8k),
        "math": ("MATH", eval_math),
        "arc": ("ARC", eval_arc),
        "mmlu_pro": ("MMLU-Pro", eval_mmlu_pro),
    }

    benchmarks = args.benchmarks.split(",")
    for bench_key in benchmarks:
        if bench_key in bench_fns:
            label, fn = bench_fns[bench_key]
            print(f"\n  {label} ({n} problems)...")
            scores = fn(sc, tokenizer, renderer, training_problems, n=n)
            result[bench_key] = scores
            print(f"  {label}: {scores['accuracy']}% acc | {scores['format_pct']}% format | {scores['avg_thinking']} think tok")

    print(f"\n  Trick questions (5)...")
    result["trick_questions"] = eval_tricks(sc, tokenizer, renderer)

    out_file = EVAL_DIR / f"{args.name}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✅ Saved: {out_file}")

    # Summary
    print(f"\n  {'='*50}")
    print(f"  SUMMARY: {args.name}")
    for bench_key in benchmarks:
        if bench_key in result:
            print(f"    {bench_key}: {result[bench_key]['accuracy']}%")
    tricks_correct = sum(1 for t in result["trick_questions"] if t["correct"])
    print(f"    tricks: {tricks_correct}/5")


if __name__ == "__main__":
    main()
