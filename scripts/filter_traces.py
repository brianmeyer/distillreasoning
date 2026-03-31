"""Filter reasoning traces for quality. Drop incorrect answers and short/repetitive traces."""

import json
import re
from pathlib import Path
from collections import Counter

INPUT_FILE = Path("data/traces_raw.jsonl")
OUTPUT_FILE = Path("data/traces_filtered.jsonl")

MIN_THINKING_TOKENS = 50


def extract_final_number(text):
    """Extract the last number from a response string."""
    # Look for numbers (including negatives and decimals)
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def extract_boxed_answer(text):
    """Extract answer from \\boxed{...} format (MATH dataset)."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def check_gsm8k_answer(response, expected):
    """Check if the model's final numeric answer matches the expected GSM8K answer."""
    try:
        expected_num = float(expected.replace(",", ""))
    except (ValueError, AttributeError):
        return None  # Can't verify, keep it

    model_num = extract_final_number(response)
    if model_num is None:
        return False
    return abs(model_num - expected_num) < 0.01


def check_arc_answer(response, expected):
    """Check if ARC multiple choice answer matches."""
    # Look for the answer letter in the response
    response_upper = response.strip().upper()
    # Check if the expected letter appears as a clear answer choice
    # Look for patterns like "The answer is A" or just the letter at the end
    patterns = [
        rf"\b{expected}\b\s*[\.\)]",       # "A." or "A)"
        rf"answer\s+is\s+{expected}\b",     # "answer is A"
        rf"correct\s+answer\s+is\s+{expected}\b",
        rf"\({expected}\)",                  # "(A)"
    ]
    for pat in patterns:
        if re.search(pat, response_upper):
            return True
    # Fallback: check if the letter is the last single capital letter
    last_caps = re.findall(r"\b([A-E])\b", response_upper)
    if last_caps and last_caps[-1] == expected:
        return True
    return None  # Uncertain


def is_repetitive(text):
    """Check if text has excessive repetition."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) < 4:
        return False
    unique_ratio = len(set(sentences)) / len(sentences)
    return unique_ratio < 0.5


def main():
    traces = []
    with open(INPUT_FILE) as f:
        for line in f:
            traces.append(json.loads(line))

    stats = Counter()
    kept = []

    for entry in traces:
        source = entry["source"]
        thinking = entry.get("thinking", "")
        response = entry.get("response", "")
        expected = entry.get("expected_answer", "")

        stats[f"{source}_total"] += 1

        # Check minimum thinking length
        thinking_tokens = len(thinking.split())
        if thinking_tokens < MIN_THINKING_TOKENS:
            stats[f"{source}_short_thinking"] += 1
            continue

        # Check for repetitive thinking
        if is_repetitive(thinking):
            stats[f"{source}_repetitive"] += 1
            continue

        # Source-specific verification
        if source == "gsm8k":
            result = check_gsm8k_answer(response, expected)
            if result is False:
                stats["gsm8k_wrong_answer"] += 1
                continue

        elif source == "math":
            # MATH answers are complex (LaTeX), harder to auto-verify
            # Just check that a boxed answer or clear answer exists in response
            if "boxed" not in response and "answer" not in response.lower():
                stats["math_no_answer"] += 1
                continue

        elif source == "arc":
            result = check_arc_answer(response, expected)
            if result is False:
                stats["arc_wrong_answer"] += 1
                continue

        # humaneval: keep all (code verification is complex)

        stats[f"{source}_kept"] += 1
        kept.append(entry)

    # Write filtered traces
    with open(OUTPUT_FILE, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    print(f"Filtering complete!")
    print(f"Input: {len(traces)} traces")
    print(f"Output: {len(kept)} traces -> {OUTPUT_FILE}")
    print()
    print("Stats:")
    for key in sorted(stats.keys()):
        print(f"  {key}: {stats[key]}")


if __name__ == "__main__":
    main()
