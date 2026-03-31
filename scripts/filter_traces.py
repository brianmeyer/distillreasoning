"""Filter reasoning traces for quality.

Quality gates (in order):
  1. Non-empty: thinking and response must exist
  2. Language: must be predominantly English, no encoding artifacts
  3. Length bounds: min 50, max 4000 thinking tokens
  4. Correctness: verify answer matches expected (source-specific)
  5. Repetition: drop degenerate loops
  6. Coherence: thinking must relate to the problem
  7. Self-contradiction: flag traces that reverse their own conclusions
  8. Reasoning quality: penalize shallow "the answer is X" without work

Writes detailed stats and examples to data/filter_report_{teacher}.json for the devlog.
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

TEACHER = sys.argv[1] if len(sys.argv) > 1 else "glm5"
INPUT_FILE = Path(f"data/traces_raw_{TEACHER}.jsonl")
OUTPUT_FILE = Path(f"data/traces_filtered_{TEACHER}.jsonl")
REPORT_FILE = Path(f"data/filter_report_{TEACHER}.json")

# Thresholds
MIN_THINKING_TOKENS = 50
MAX_THINKING_TOKENS = 4000   # Cap very long traces — diminishing signal
MIN_RESPONSE_TOKENS = 5      # Response must actually say something
REPETITION_THRESHOLD = 0.4   # Tighter than before (was 0.5)
MIN_STEP_KEYWORDS = 2        # Must show structured reasoning
MAX_NONSENSE_RATIO = 0.15    # Max fraction of non-ASCII or garbled chars


# ── Helper Functions ───────────────────────────────────────────────────────────

def extract_final_number(text):
    """Extract the last number from response text."""
    numbers = re.findall(r"-?\d+\.?\d*", text.replace(",", ""))
    return float(numbers[-1]) if numbers else None


def extract_boxed(text):
    """Extract answer from \\boxed{...}."""
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    return match.group(1).strip() if match else None


def check_gsm8k_answer(response, expected):
    """Verify GSM8K numeric answer."""
    try:
        expected_num = float(expected.replace(",", ""))
    except (ValueError, AttributeError):
        return None  # Can't verify
    model_num = extract_final_number(response)
    if model_num is None:
        return False
    return abs(model_num - expected_num) < 0.01


def check_arc_answer(response, expected):
    """Verify ARC multiple choice answer."""
    response_upper = response.strip().upper()
    patterns = [
        rf"answer\s+is\s+\(?{expected}\)?\b",
        rf"correct\s+(?:answer|option)\s+is\s+\(?{expected}\)?\b",
        rf"\({expected}\)",
        rf"\b{expected}\b\s*[\.\)]",
    ]
    for pat in patterns:
        if re.search(pat, response_upper):
            return True
    # Fallback: last standalone capital letter
    last_caps = re.findall(r"\b([A-E])\b", response_upper)
    if last_caps and last_caps[-1] == expected:
        return True
    return None  # Uncertain — keep it


def check_math_answer(response, expected):
    """Verify MATH answer. Expected often contains \\boxed{} in the solution."""
    # Extract boxed answer from expected solution
    expected_boxed = extract_boxed(expected)
    model_boxed = extract_boxed(response)

    if expected_boxed and model_boxed:
        # Normalize whitespace and compare
        return expected_boxed.strip() == model_boxed.strip()

    # If no boxed format, check if response mentions an answer at all
    if "answer" not in response.lower() and "boxed" not in response:
        return False

    return None  # Can't verify precisely — keep it


def check_humaneval_code(response):
    """Basic code quality check for HumanEval traces."""
    # Must contain actual Python code (def, return, or indented blocks)
    has_code = bool(re.search(r'(def\s+\w+|return\s+|if\s+.*:)', response))
    if not has_code:
        return False
    # Must not be just the prompt echoed back
    if len(response.strip()) < 50:
        return False
    return True


# ── Quality Checks ─────────────────────────────────────────────────────────────

def is_garbled(text):
    """Check for encoding artifacts, excessive non-ASCII, or gibberish."""
    if not text:
        return True
    # Count non-ASCII chars (excluding common math/unicode)
    non_ascii = sum(1 for c in text if ord(c) > 127 and c not in '×÷±≈≠≤≥∞π√∑∏∫∂∇αβγδεθλμσφωΔ')
    ratio = non_ascii / len(text) if text else 0
    if ratio > MAX_NONSENSE_RATIO:
        return True
    # Check for common encoding artifacts
    if any(artifact in text for artifact in ['â€™', 'â€"', 'Ã©', 'Ã¼', '\x00', '\\u0000']):
        return True
    return False


def is_repetitive(text):
    """Detect degenerate repetition loops."""
    sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if s.strip() and len(s.strip()) > 10]
    if len(sentences) < 4:
        return False
    unique_ratio = len(set(sentences)) / len(sentences)
    if unique_ratio < REPETITION_THRESHOLD:
        return True

    # Also check for repeated phrases (3-grams)
    words = text.lower().split()
    if len(words) > 20:
        trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        trigram_counts = Counter(trigrams)
        most_common_count = trigram_counts.most_common(1)[0][1] if trigram_counts else 0
        # If any trigram appears more than 10% of all trigrams, suspicious
        if most_common_count > len(trigrams) * 0.1 and most_common_count > 5:
            return True

    return False


def has_structured_reasoning(thinking):
    """Check that thinking shows actual step-by-step work, not just a flat statement."""
    step_indicators = [
        r'step\s*\d', r'first[,\s]', r'second[,\s]', r'third[,\s]',
        r'next[,\s]', r'then[,\s]', r'therefore', r'because',
        r'let\s+(me|us|\'s)', r'so\s+', r'this\s+means',
        r'we\s+(can|need|have|know|get|see)',
        r'if\s+', r'since\s+', r'given\s+',
        r'\d+\s*[+\-*/×÷=]', r'=\s*\d+',
    ]
    matches = sum(1 for pat in step_indicators if re.search(pat, thinking.lower()))
    return matches >= MIN_STEP_KEYWORDS


def has_self_contradiction(thinking):
    """Detect cases where the model reverses its own conclusion."""
    contradiction_patterns = [
        r'wait[,.]?\s*(that\'s|this is|that is)\s*(wrong|incorrect|not right)',
        r'actually[,.]?\s*(that\'s|this is)\s*(wrong|incorrect)',
        r'no[,.]?\s*(that\'s|this is)\s*(wrong|incorrect|not)',
        r'I\s*(was|am)\s*wrong',
        r'let me\s*(re-?do|re-?calculate|start over|try again)',
    ]
    contradiction_count = sum(1 for pat in contradiction_patterns if re.search(pat, thinking.lower()))
    # One self-correction is fine (shows good reasoning). Multiple = confused.
    return contradiction_count >= 3


def thinking_matches_problem(thinking, problem):
    """Basic coherence check: does the thinking reference the problem's content?"""
    # Extract key numbers and nouns from the problem
    problem_numbers = set(re.findall(r'\b\d+\b', problem))
    thinking_numbers = set(re.findall(r'\b\d+\b', thinking))

    if problem_numbers:
        # At least some problem numbers should appear in thinking
        overlap = problem_numbers & thinking_numbers
        if len(overlap) == 0 and len(problem_numbers) > 1:
            return False

    # Extract key words (>4 chars) from problem
    problem_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', problem))
    thinking_words = set(w.lower() for w in re.findall(r'\b[a-zA-Z]{5,}\b', thinking[:500]))

    if problem_words:
        overlap = problem_words & thinking_words
        if len(overlap) < min(2, len(problem_words)):
            return False

    return True


# ── Main Filter Pipeline ──────────────────────────────────────────────────────

def main():
    traces = []
    with open(INPUT_FILE) as f:
        for line in f:
            traces.append(json.loads(line))

    stats = Counter()
    dropped = defaultdict(list)
    kept_examples = defaultdict(list)
    kept = []

    for entry in traces:
        source = entry["source"]
        thinking = entry.get("thinking", "")
        response = entry.get("response", "")
        expected = entry.get("expected_answer", "")
        problem = entry.get("problem", "")

        stats[f"{source}_total"] += 1
        thinking_tokens = len(thinking.split())
        response_tokens = len(response.split())
        drop_reason = None

        # ── Gate 1: Non-empty ──
        if not thinking or not response:
            drop_reason = "empty"

        # ── Gate 2: Language quality ──
        elif is_garbled(thinking) or is_garbled(response):
            drop_reason = "garbled"

        # ── Gate 3: Length bounds ──
        elif thinking_tokens < MIN_THINKING_TOKENS:
            drop_reason = "short_thinking"
        elif thinking_tokens > MAX_THINKING_TOKENS:
            drop_reason = "too_long"
        elif response_tokens < MIN_RESPONSE_TOKENS:
            drop_reason = "short_response"

        # ── Gate 4: Correctness (source-specific) ──
        elif source == "gsm8k":
            result = check_gsm8k_answer(response, expected)
            if result is False:
                drop_reason = "wrong_answer"
        elif source == "math":
            result = check_math_answer(response, expected)
            if result is False:
                drop_reason = "wrong_answer"
        elif source == "arc":
            result = check_arc_answer(response, expected)
            if result is False:
                drop_reason = "wrong_answer"
        elif source == "humaneval":
            if not check_humaneval_code(response):
                drop_reason = "bad_code"

        # ── Gate 5: Repetition ──
        if not drop_reason and is_repetitive(thinking):
            drop_reason = "repetitive"

        # ── Gate 6: Coherence ──
        if not drop_reason and not thinking_matches_problem(thinking, problem):
            drop_reason = "incoherent"

        # ── Gate 7: Self-contradiction ──
        if not drop_reason and has_self_contradiction(thinking):
            drop_reason = "self_contradictory"

        # ── Gate 8: Structured reasoning ──
        if not drop_reason and not has_structured_reasoning(thinking):
            drop_reason = "no_reasoning_structure"

        # ── Decision ──
        if drop_reason:
            stats[f"{source}_dropped_{drop_reason}"] += 1
            if len(dropped[drop_reason]) < 3:
                example = {
                    "id": entry["id"],
                    "source": source,
                    "problem": problem[:200],
                    "thinking_tokens": thinking_tokens,
                    "reason": drop_reason,
                }
                if drop_reason == "wrong_answer":
                    example["expected"] = str(expected)[:100]
                    example["model_response_tail"] = response[-200:]
                if drop_reason in ("short_thinking", "no_reasoning_structure"):
                    example["thinking_preview"] = thinking[:300]
                if drop_reason == "repetitive":
                    example["thinking_preview"] = thinking[:400]
                if drop_reason == "too_long":
                    example["thinking_tokens"] = thinking_tokens
                dropped[drop_reason].append(example)
            continue

        # ── Kept ──
        stats[f"{source}_kept"] += 1
        entry["thinking_tokens"] = thinking_tokens
        kept.append(entry)

        if len(kept_examples[source]) < 2:
            kept_examples[source].append({
                "id": entry["id"],
                "problem": problem[:300],
                "thinking_tokens": thinking_tokens,
                "thinking_preview": thinking[:500],
                "response_preview": response[:300],
            })

    # ── Stats ──
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

    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")

    # Write report
    report = {
        "teacher": TEACHER,
        "summary": {
            "total_input": len(traces),
            "total_kept": len(kept),
            "total_dropped": len(traces) - len(kept),
            "keep_rate_pct": round(100 * len(kept) / len(traces), 1) if traces else 0,
        },
        "by_source": {},
        "thinking_length_distribution": length_stats,
        "gates": [
            "1. Non-empty (thinking + response exist)",
            "2. Language quality (no garbled/encoding artifacts)",
            "3. Length bounds (50-4000 thinking tokens, 5+ response tokens)",
            "4. Correctness (answer verification per source)",
            "5. Repetition (sentence + trigram deduplication)",
            "6. Coherence (thinking references problem content)",
            "7. Self-contradiction (max 2 self-corrections)",
            "8. Structured reasoning (step indicators present)",
        ],
        "drop_examples": dict(dropped),
        "kept_examples": dict(kept_examples),
    }

    all_drop_reasons = set()
    for source in ["gsm8k", "math", "arc", "humaneval"]:
        total = stats.get(f"{source}_total", 0)
        k = stats.get(f"{source}_kept", 0)
        reasons = {}
        for stat_key, v in stats.items():
            if stat_key.startswith(f"{source}_dropped_"):
                reason_name = stat_key.replace(f"{source}_dropped_", "")
                reasons[reason_name] = v
                all_drop_reasons.add(reason_name)
        report["by_source"][source] = {
            "total": total,
            "kept": k,
            "dropped": total - k,
            "keep_rate_pct": round(100 * k / total, 1) if total else 0,
            "drop_reasons": reasons,
        }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FILTER RESULTS — {TEACHER.upper()}")
    print(f"{'='*60}")
    print(f"Input:   {len(traces)}")
    print(f"Kept:    {len(kept)} ({report['summary']['keep_rate_pct']}%)")
    print(f"Dropped: {len(traces) - len(kept)}")
    print()
    print(f"Quality gates applied:")
    for gate in report["gates"]:
        print(f"  {gate}")
    print()
    for source, d in report["by_source"].items():
        if d["total"] > 0:
            print(f"  {source:12s}: {d['kept']:4d}/{d['total']:4d} kept ({d['keep_rate_pct']}%)")
            for reason, count in d["drop_reasons"].items():
                print(f"               - {count} dropped: {reason}")
    print()
    if length_stats:
        print(f"Thinking length (tokens): min={length_stats['min']} "
              f"p25={length_stats['p25']} median={length_stats['median']} "
              f"p75={length_stats['p75']} max={length_stats['max']} "
              f"mean={length_stats['mean']}")
    print()

    # Aggregate drop reasons
    print("Drop reasons across all sources:")
    for reason in sorted(all_drop_reasons):
        total_drops = sum(
            stats.get(f"{s}_dropped_{reason}", 0)
            for s in ["gsm8k", "math", "arc", "humaneval"]
        )
        print(f"  {reason:25s}: {total_drops}")

    print(f"\nFull report: {REPORT_FILE}")
    print(f"Filtered traces: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
