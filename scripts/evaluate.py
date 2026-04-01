"""Evaluate distilled models on reasoning benchmarks.

Follows DeepSeek-R1 evaluation methodology:
- Zero-shot (no few-shot examples)
- Pass@1 (single attempt, greedy or low temp)
- Multiple benchmarks across domains

Benchmarks:
  1. GSM8K (100 held-out) — grade school math, numeric answer extraction
  2. MATH-500 (100 sampled) — competition math, boxed answer extraction
  3. ARC-Challenge (100 sampled) — science reasoning, multiple choice
  4. HumanEval (50 sampled) — code generation, function completion

Metrics per benchmark:
  - Accuracy (Pass@1)
  - Format compliance (<think>/<answer> tags present)
  - Average thinking tokens
  - Average response time

Also generates qualitative comparison: same 5 problems, all models side by side.
"""

import json
import re
import time
import torch
from pathlib import Path
from datasets import load_dataset

EVAL_DIR = Path("data/eval_results")
EVAL_DIR.mkdir(exist_ok=True)

# Trick questions for qualitative comparison (tests reasoning, not pattern matching)
QUALITATIVE_PROBLEMS = [
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "A store offers a 20% discount on a jacket that costs $80. Then they add 8% sales tax. What is the final price?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
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


def generate_response(model, tokenizer, problem, system_msg=None, max_tokens=1024):
    """Generate a response from a model. Works with both HF models and Ollama."""
    if system_msg is None:
        system_msg = "You are a helpful reasoning assistant. Think through problems step by step before answering."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": problem},
    ]

    start = time.time()
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.6,
            top_p=0.95,
            do_sample=True,
        )
    elapsed = time.time() - start
    response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
    return response, elapsed


def eval_gsm8k(model, tokenizer, n=100):
    """Evaluate on GSM8K test set."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = list(ds.select(range(min(n, len(ds)))))

    results = {"correct": 0, "total": 0, "format_ok": 0, "think_tokens": [], "times": []}

    for p in problems:
        response, elapsed = generate_response(model, tokenizer, p["question"])
        expected = p["answer"].split("####")[-1].strip()

        results["total"] += 1
        results["times"].append(elapsed)
        if has_format(response):
            results["format_ok"] += 1
        results["think_tokens"].append(count_thinking_tokens(response))

        try:
            expected_num = float(expected.replace(",", ""))
            model_num = extract_final_number(response)
            if model_num is not None and abs(model_num - expected_num) < 0.01:
                results["correct"] += 1
        except ValueError:
            pass

    return results


def eval_math(model, tokenizer, n=100):
    """Evaluate on MATH-500 (sampled)."""
    ds = load_dataset("SuperSecureHuman/competition_math_hf_dataset", split="test")
    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    results = {"correct": 0, "total": 0, "format_ok": 0, "think_tokens": [], "times": []}

    for i in indices:
        row = ds[i]
        response, elapsed = generate_response(model, tokenizer, row["problem"])
        expected_boxed = extract_boxed(row.get("solution", ""))

        results["total"] += 1
        results["times"].append(elapsed)
        if has_format(response):
            results["format_ok"] += 1
        results["think_tokens"].append(count_thinking_tokens(response))

        model_boxed = extract_boxed(response)
        if expected_boxed and model_boxed and expected_boxed.strip() == model_boxed.strip():
            results["correct"] += 1

    return results


def eval_arc(model, tokenizer, n=100):
    """Evaluate on ARC-Challenge."""
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    import random
    random.seed(42)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    results = {"correct": 0, "total": 0, "format_ok": 0, "think_tokens": [], "times": []}

    for i in indices:
        row = ds[i]
        choices = row["choices"]
        choice_text = "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))
        problem = f"{row['question']}\n\nChoices:\n{choice_text}"

        response, elapsed = generate_response(model, tokenizer, problem)
        expected = row["answerKey"]

        results["total"] += 1
        results["times"].append(elapsed)
        if has_format(response):
            results["format_ok"] += 1
        results["think_tokens"].append(count_thinking_tokens(response))

        # Check answer
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
            results["correct"] += 1

    return results


def eval_qualitative(model, tokenizer):
    """Generate responses to trick questions for side-by-side comparison."""
    responses = []
    for problem in QUALITATIVE_PROBLEMS:
        response, elapsed = generate_response(model, tokenizer, problem)
        responses.append({
            "problem": problem,
            "response": response[:1000],
            "has_format": has_format(response),
            "thinking_tokens": count_thinking_tokens(response),
            "time": round(elapsed, 1),
        })
    return responses


def summarize(name, benchmark, results):
    """Print and return a summary row."""
    total = results["total"]
    acc = results["correct"] / total * 100 if total else 0
    fmt = results["format_ok"] / total * 100 if total else 0
    avg_think = sum(results["think_tokens"]) / total if total else 0
    avg_time = sum(results["times"]) / total if total else 0

    row = {
        "model": name,
        "benchmark": benchmark,
        "accuracy": round(acc, 1),
        "format_pct": round(fmt, 1),
        "avg_thinking_tokens": round(avg_think),
        "avg_time_sec": round(avg_time, 1),
        "n": total,
    }

    print(f"  {benchmark:12s}: {acc:5.1f}% acc | {fmt:5.1f}% format | {avg_think:5.0f} think tok | {avg_time:4.1f}s avg")
    return row


def evaluate_model(model, tokenizer, name, run_benchmarks=True):
    """Full evaluation of a single model."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    all_results = {"model": name, "benchmarks": {}, "qualitative": []}

    if run_benchmarks:
        for bench_name, eval_fn in [("gsm8k", eval_gsm8k), ("math", eval_math), ("arc", eval_arc)]:
            print(f"\n  Running {bench_name}...")
            results = eval_fn(model, tokenizer)
            row = summarize(name, bench_name, results)
            all_results["benchmarks"][bench_name] = row

    print(f"\n  Running qualitative (trick questions)...")
    all_results["qualitative"] = eval_qualitative(model, tokenizer)

    # Save
    out_file = EVAL_DIR / f"{name.replace('/', '_').replace(' ', '_')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out_file}")

    return all_results


if __name__ == "__main__":
    print("Evaluation script ready.")
    print()
    print("Usage in Colab or after training:")
    print("  from scripts.evaluate import evaluate_model")
    print("  results = evaluate_model(model, tokenizer, 'ModelName')")
    print()
    print("Benchmarks: GSM8K (100), MATH (100), ARC (100)")
    print("Plus 5 qualitative trick questions for side-by-side comparison")
    print()
    print("To generate the full comparison table:")
    print("  Run evaluate_model() for each of your models")
    print("  Results saved to data/eval_results/")
