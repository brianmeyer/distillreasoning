"""Evaluate all models on Tinker using the sampling API.

Benchmarks:
  - GSM8K (50 problems) — grade school math, numeric answer
  - Trick questions (5) — qualitative side-by-side

Models evaluated:
  - Base Qwen3.5-4B (no LoRA)
  - 4B + GLM-5 (final checkpoint)
  - 4B + Kimi (final checkpoint)
  - 4B + Combined (final checkpoint)

Usage: TINKER_API_KEY=xxx python scripts/eval_tinker.py
"""

import tinker
from tinker import types
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from datasets import load_dataset
import json, re, time, random
from pathlib import Path

random.seed(42)

EVAL_DIR = Path("data/eval_results")
EVAL_DIR.mkdir(exist_ok=True)

MODEL = "Qwen/Qwen3.5-4B"
GSM8K_N = 50  # Keep eval manageable on Tinker credits

SYSTEM_MSG = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRICK_QUESTIONS = [
    "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "A store offers a 20% discount on a jacket that costs $80. Then they add 8% sales tax. What is the final price?",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
]

CHECKPOINTS = [
    {"name": "base", "path": None},
    {"name": "4b-glm5", "path": "sampler_weights/qwen35-4b-glm5-final"},
    {"name": "4b-kimi", "path": "sampler_weights/qwen35-4b-kimi-final"},
    {"name": "4b-combined", "path": "sampler_weights/qwen35-4b-combined-final"},
]


def extract_final_number(text):
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def has_format(text):
    return "<think>" in text and "</think>" in text


def count_thinking_tokens(text):
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return len(match.group(1).split()) if match else 0


def sample_from_model(sampling_client, tokenizer, renderer, problem, max_tokens=1024):
    """Generate a response using Tinker sampling."""
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": problem},
    ]
    prompt = renderer.build_generation_prompt(messages)
    stop_sequences = renderer.get_stop_sequences()

    params = types.SamplingParams(
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.95,
        stop=stop_sequences,
    )

    start = time.time()
    result = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1).result()
    elapsed = time.time() - start

    if result.sequences:
        response = tokenizer.decode(result.sequences[0].tokens)
    else:
        response = ""

    return response, elapsed


def eval_gsm8k(sampling_client, tokenizer, renderer, n=GSM8K_N):
    """Evaluate on GSM8K."""
    ds = load_dataset("openai/gsm8k", "main", split="test")
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    correct = 0
    format_ok = 0
    think_tokens = []
    total = 0

    for i in indices:
        row = ds[i]
        response, elapsed = sample_from_model(sampling_client, tokenizer, renderer, row["question"])
        expected = row["answer"].split("####")[-1].strip()

        total += 1
        if has_format(response):
            format_ok += 1
        think_tokens.append(count_thinking_tokens(response))

        try:
            expected_num = float(expected.replace(",", ""))
            model_num = extract_final_number(response)
            if model_num is not None and abs(model_num - expected_num) < 0.01:
                correct += 1
        except ValueError:
            pass

        if total % 10 == 0:
            print(f"    GSM8K: {total}/{n} done, {correct}/{total} correct so far")

    return {
        "accuracy": round(100 * correct / total, 1),
        "format_pct": round(100 * format_ok / total, 1),
        "avg_thinking": round(sum(think_tokens) / total) if total else 0,
        "n": total,
        "correct": correct,
    }


def eval_tricks(sampling_client, tokenizer, renderer):
    """Qualitative trick question evaluation."""
    results = []
    for q in TRICK_QUESTIONS:
        response, elapsed = sample_from_model(sampling_client, tokenizer, renderer, q)
        results.append({
            "problem": q,
            "response": response[:1500],
            "has_format": has_format(response),
            "thinking_tokens": count_thinking_tokens(response),
            "time": round(elapsed, 1),
        })
    return results


def evaluate_model(service, tokenizer, renderer, checkpoint):
    """Full evaluation of one model."""
    name = checkpoint["name"]
    path = checkpoint["path"]

    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    if path:
        print(f"  Loading checkpoint: {path}")
        sampling_client = service.create_sampling_client(model_path=path)
    else:
        print(f"  Loading base model (no LoRA)")
        sampling_client = service.create_sampling_client(base_model=MODEL)

    print(f"  Running GSM8K ({GSM8K_N} problems)...")
    gsm8k = eval_gsm8k(sampling_client, tokenizer, renderer)
    print(f"  GSM8K: {gsm8k['accuracy']}% accuracy, {gsm8k['format_pct']}% format, {gsm8k['avg_thinking']} avg think tokens")

    print(f"  Running trick questions...")
    tricks = eval_tricks(sampling_client, tokenizer, renderer)

    result = {
        "model": name,
        "checkpoint": path,
        "gsm8k": gsm8k,
        "trick_questions": tricks,
    }

    out_file = EVAL_DIR / f"{name}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_file}")

    return result


def main():
    print("Setting up Tinker...")
    service = tinker.ServiceClient()
    tokenizer = tokenizer_utils.get_tokenizer(MODEL)
    renderer = get_renderer(model_info.get_recommended_renderer_name(MODEL), tokenizer)

    all_results = []
    for checkpoint in CHECKPOINTS:
        result = evaluate_model(service, tokenizer, renderer, checkpoint)
        all_results.append(result)

    # Print comparison table
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':20s} {'GSM8K':>8s} {'Format':>8s} {'Think':>8s}")
    print(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8}")
    for r in all_results:
        g = r["gsm8k"]
        print(f"{r['model']:20s} {g['accuracy']:>7.1f}% {g['format_pct']:>7.1f}% {g['avg_thinking']:>7d}")

    # Print trick question comparison
    print(f"\n{'='*60}")
    print("TRICK QUESTIONS — Side by Side")
    print(f"{'='*60}")
    for i, q in enumerate(TRICK_QUESTIONS):
        print(f"\nQ: {q}")
        for r in all_results:
            trick = r["trick_questions"][i]
            answer_preview = trick["response"][:200].replace("\n", " ")
            fmt = "✓" if trick["has_format"] else "✗"
            print(f"  [{r['model']:15s}] [{fmt}] {answer_preview}...")

    # Save combined results
    with open(EVAL_DIR / "comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull comparison: {EVAL_DIR}/comparison.json")


if __name__ == "__main__":
    main()
