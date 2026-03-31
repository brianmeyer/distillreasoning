"""Filter reasoning traces for quality. Drop incorrect answers and short/repetitive traces.

Writes detailed stats and examples to data/filter_report.json for the devlog.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import sys

# Accept teacher name as argument: python filter_traces.py [glm5|kimi]
TEACHER = sys.argv[1] if len(sys.argv) > 1 else "glm5"
INPUT_FILE = Path(f"data/traces_raw_{TEACHER}.jsonl")
OUTPUT_FILE = Path(f"data/traces_filtered_{TEACHER}.jsonl")
REPORT_FILE = Path(f"data/filter_report_{TEACHER}.json")

MIN_THINKING_TOKENS = 50


def extract_final_number(text):
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def check_gsm8k_answer(response, expected):
    try:
        expected_num = float(expected.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    model_num = extract_final_number(response)
    if model_num is None:
        return False
    return abs(model_num - expected_num) < 0.01


def check_arc_answer(response, expected):
    response_upper = response.strip().upper()
    patterns = [
        rf"\b{expected}\b\s*[\.\)]",
        rf"answer\s+is\s+{expected}\b",
        rf"correct\s+answer\s+is\s+{expected}\b",
        rf"\({expected}\)",
    ]
    for pat in patterns:
        if re.search(pat, response_upper):
            return True
    last_caps = re.findall(r"\b([A-E])\b", response_upper)
    if last_caps and last_caps[-1] == expected:
        return True
    return None


def is_repetitive(text):
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
    # Track drop reasons with examples
    dropped = defaultdict(list)  # reason -> list of examples
    kept_examples = defaultdict(list)  # source -> list of examples (first 2 each)
    kept = []

    for entry in traces:
        source = entry["source"]
        thinking = entry.get("thinking", "")
        response = entry.get("response", "")
        expected = entry.get("expected_answer", "")

        stats[f"{source}_total"] += 1
        thinking_tokens = len(thinking.split())

        # Check: minimum thinking length
        if thinking_tokens < MIN_THINKING_TOKENS:
            stats[f"{source}_dropped_short_thinking"] += 1
            if len(dropped["short_thinking"]) < 3:
                dropped["short_thinking"].append({
                    "id": entry["id"],
                    "source": source,
                    "problem": entry["problem"][:200],
                    "thinking_tokens": thinking_tokens,
                    "thinking_preview": thinking[:200],
                })
            continue

        # Check: repetitive thinking
        if is_repetitive(thinking):
            stats[f"{source}_dropped_repetitive"] += 1
            if len(dropped["repetitive"]) < 3:
                dropped["repetitive"].append({
                    "id": entry["id"],
                    "source": source,
                    "problem": entry["problem"][:200],
                    "thinking_preview": thinking[:300],
                })
            continue

        # Check: source-specific answer verification
        if source == "gsm8k":
            result = check_gsm8k_answer(response, expected)
            if result is False:
                stats["gsm8k_dropped_wrong_answer"] += 1
                if len(dropped["wrong_answer"]) < 3:
                    dropped["wrong_answer"].append({
                        "id": entry["id"],
                        "source": source,
                        "problem": entry["problem"][:200],
                        "expected": expected,
                        "model_response_tail": response[-200:],
                    })
                continue

        elif source == "math":
            if "boxed" not in response and "answer" not in response.lower():
                stats["math_dropped_no_answer"] += 1
                if len(dropped["no_answer"]) < 3:
                    dropped["no_answer"].append({
                        "id": entry["id"],
                        "source": source,
                        "problem": entry["problem"][:200],
                        "response_preview": response[:300],
                    })
                continue

        elif source == "arc":
            result = check_arc_answer(response, expected)
            if result is False:
                stats["arc_dropped_wrong_answer"] += 1
                if len(dropped["wrong_answer"]) < 3:
                    dropped["wrong_answer"].append({
                        "id": entry["id"],
                        "source": source,
                        "problem": entry["problem"][:200],
                        "expected": expected,
                        "model_response_tail": response[-200:],
                    })
                continue

        # Kept
        stats[f"{source}_kept"] += 1
        entry["thinking_tokens"] = thinking_tokens
        kept.append(entry)

        # Save a couple of examples per source for the devlog
        if len(kept_examples[source]) < 2:
            kept_examples[source].append({
                "id": entry["id"],
                "problem": entry["problem"][:300],
                "thinking_tokens": thinking_tokens,
                "thinking_preview": thinking[:500],
                "response_preview": response[:300],
            })

    # Thinking length distribution
    thinking_lengths = [e["thinking_tokens"] for e in kept]
    if thinking_lengths:
        thinking_lengths.sort()
        n = len(thinking_lengths)
        length_stats = {
            "min": thinking_lengths[0],
            "p25": thinking_lengths[n // 4],
            "median": thinking_lengths[n // 2],
            "p75": thinking_lengths[3 * n // 4],
            "max": thinking_lengths[-1],
            "mean": int(sum(thinking_lengths) / n),
        }
    else:
        length_stats = {}

    # Write filtered traces
    with open(OUTPUT_FILE, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    # Write report
    report = {
        "summary": {
            "total_input": len(traces),
            "total_kept": len(kept),
            "total_dropped": len(traces) - len(kept),
            "keep_rate_pct": round(100 * len(kept) / len(traces), 1) if traces else 0,
        },
        "by_source": {},
        "thinking_length_distribution": length_stats,
        "drop_examples": dict(dropped),
        "kept_examples": dict(kept_examples),
    }

    for source in ["gsm8k", "math", "arc", "humaneval"]:
        total = stats.get(f"{source}_total", 0)
        k = stats.get(f"{source}_kept", 0)
        report["by_source"][source] = {
            "total": total,
            "kept": k,
            "dropped": total - k,
            "keep_rate_pct": round(100 * k / total, 1) if total else 0,
            "drop_reasons": {
                k2.replace(f"{source}_dropped_", ""): v
                for k2, v in stats.items()
                if k2.startswith(f"{source}_dropped_")
            },
        }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print(f"FILTER RESULTS")
    print(f"{'='*50}")
    print(f"Input:   {len(traces)}")
    print(f"Kept:    {len(kept)} ({report['summary']['keep_rate_pct']}%)")
    print(f"Dropped: {len(traces) - len(kept)}")
    print()
    for source, d in report["by_source"].items():
        if d["total"] > 0:
            print(f"  {source:12s}: {d['kept']:4d}/{d['total']:4d} kept ({d['keep_rate_pct']}%)")
            for reason, count in d["drop_reasons"].items():
                print(f"               - {count} dropped: {reason}")
    print()
    print(f"Thinking length (tokens): min={length_stats.get('min','?')} "
          f"median={length_stats.get('median','?')} "
          f"max={length_stats.get('max','?')} "
          f"mean={length_stats.get('mean','?')}")
    print(f"\nFull report: {REPORT_FILE}")
    print(f"Filtered traces: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
