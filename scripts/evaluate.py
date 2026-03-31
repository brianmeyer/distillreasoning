"""Evaluate model on held-out GSM8K problems. Run in Colab after training."""

import json
import re
import torch
from datasets import load_dataset


def extract_final_number(text):
    """Extract the last number from response text."""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def has_format(text):
    """Check if response uses <think>/<answer> tags."""
    return "<think>" in text and "</think>" in text and "<answer>" in text and "</answer>" in text


def count_thinking_tokens(text):
    """Count tokens in the thinking section."""
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        return len(match.group(1).split())
    return 0


def evaluate_model(model, tokenizer, problems, model_name="model"):
    """Run evaluation on a list of GSM8K problems."""
    results = {
        "correct": 0,
        "total": 0,
        "format_compliant": 0,
        "total_thinking_tokens": 0,
    }

    for i, problem in enumerate(problems):
        question = problem["question"]
        expected = problem["answer"].split("####")[-1].strip()

        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant. Think through problems step by step before answering."},
            {"role": "user", "content": question},
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)

        results["total"] += 1

        # Check format
        if has_format(response):
            results["format_compliant"] += 1

        # Count thinking tokens
        results["total_thinking_tokens"] += count_thinking_tokens(response)

        # Check correctness
        try:
            expected_num = float(expected.replace(",", ""))
            model_num = extract_final_number(response)
            if model_num is not None and abs(model_num - expected_num) < 0.01:
                results["correct"] += 1
        except ValueError:
            pass

        if (i + 1) % 10 == 0:
            print(f"  [{model_name}] {i+1}/{len(problems)} done...")

    return results


def print_results(name, results):
    """Print evaluation results for a model."""
    total = results["total"]
    print(f"\n{name}:")
    print(f"  Accuracy: {results['correct']}/{total} ({100*results['correct']/total:.1f}%)")
    print(f"  Format compliance: {results['format_compliant']}/{total} ({100*results['format_compliant']/total:.1f}%)")
    avg_think = results['total_thinking_tokens'] / total if total > 0 else 0
    print(f"  Avg thinking tokens: {avg_think:.0f}")


if __name__ == "__main__":
    # This is meant to be run in Colab after training
    # Load 100 GSM8K test problems
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    eval_problems = list(gsm8k.select(range(100)))

    print(f"Evaluating on {len(eval_problems)} GSM8K test problems\n")

    # You'd load each model variant here:
    # 1. Base Qwen3.5-4B
    # 2. SFT checkpoint
    # 3. SFT+GRPO checkpoint (if available)

    print("To use this script in Colab:")
    print("1. Load each model with FastLanguageModel.from_pretrained()")
    print("2. Call evaluate_model(model, tokenizer, eval_problems, 'Model Name')")
    print("3. Call print_results('Model Name', results)")
