"""Format filtered traces into SFT chat format with stratified train/val/test split.

Stratified = each domain (gsm8k, math, arc, humaneval) gets proportional
representation in every split. No domain disappears from the test set.

Writes detailed formatting report to data/format_report_{teacher}.json for the devlog.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

import sys

TEACHER = sys.argv[1] if len(sys.argv) > 1 else "glm5"
INPUT_FILE = Path(f"data/traces_filtered_{TEACHER}.jsonl")
TRAIN_FILE = Path(f"data/train_{TEACHER}.jsonl")
VAL_FILE = Path(f"data/validation_{TEACHER}.jsonl")
TEST_FILE = Path(f"data/test_{TEACHER}.jsonl")
REPORT_FILE = Path(f"data/format_report_{TEACHER}.json")

SYSTEM_MESSAGE = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
# Test gets the rest (0.10)


def format_entry(entry):
    """Convert a trace entry to chat message format with <think>/<answer> tags."""
    thinking = entry.get("thinking", "").strip()
    response = entry.get("response", "").strip()
    assistant_content = f"<think>\n{thinking}\n</think>\n\n<answer>\n{response}\n</answer>"
    return {
        "source": entry.get("source", "unknown"),
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def count_tokens_approx(text):
    return int(len(text.split()) / 0.75)


def stratified_split(entries, train_ratio, val_ratio):
    """Split entries into train/val/test with proportional source representation."""
    by_source = defaultdict(list)
    for entry in entries:
        by_source[entry["source"]].append(entry)

    train, val, test = [], [], []
    split_report = {}

    for source, items in sorted(by_source.items()):
        random.shuffle(items)
        n = len(items)
        train_end = max(1, int(n * train_ratio))
        val_end = train_end + max(1, int(n * val_ratio))

        src_train = items[:train_end]
        src_val = items[train_end:val_end]
        src_test = items[val_end:]

        # Ensure test has at least 1
        if not src_test and src_val:
            src_test = [src_val.pop()]
        elif not src_test and src_train:
            src_test = [src_train.pop()]

        train.extend(src_train)
        val.extend(src_val)
        test.extend(src_test)

        split_report[source] = {
            "total": n,
            "train": len(src_train),
            "val": len(src_val),
            "test": len(src_test),
        }

    # Final shuffle within each split (mix sources)
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test, split_report


def main():
    entries = []
    with open(INPUT_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    formatted = [format_entry(e) for e in entries]

    # Stratified split
    train, val, test, split_report = stratified_split(formatted, TRAIN_RATIO, VAL_RATIO)

    # Remove source field before writing (not needed for training)
    def strip_source(items):
        for item in items:
            item_copy = {"messages": item["messages"]}
            yield item_copy

    # Write splits
    for data, path in [(train, TRAIN_FILE), (val, VAL_FILE), (test, TEST_FILE)]:
        with open(path, "w") as f:
            for entry in strip_source(data):
                f.write(json.dumps(entry) + "\n")

    # Token length stats
    total_tokens, thinking_tokens, answer_tokens = [], [], []
    for entry in formatted:
        assistant = entry["messages"][2]["content"]
        total_tokens.append(count_tokens_approx(assistant))
        think_start = assistant.find("<think>\n") + len("<think>\n")
        think_end = assistant.find("\n</think>")
        answer_start = assistant.find("<answer>\n") + len("<answer>\n")
        answer_end = assistant.find("\n</answer>")
        if think_end > think_start:
            thinking_tokens.append(count_tokens_approx(assistant[think_start:think_end]))
        if answer_end > answer_start:
            answer_tokens.append(count_tokens_approx(assistant[answer_start:answer_end]))

    def dist(vals):
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "min": s[0], "p25": s[n//4], "median": s[n//2],
            "p75": s[3*n//4], "max": s[-1], "mean": int(sum(s)/n)
        }

    # Grab examples per source for devlog
    samples = []
    seen_sources = set()
    for e in entries:
        if e["source"] not in seen_sources:
            seen_sources.add(e["source"])
            fmt = format_entry(e)
            samples.append({
                "source": e["source"],
                "problem": e["problem"][:300],
                "expected_answer": e.get("expected_answer", "")[:100],
                "formatted_assistant_preview": fmt["messages"][2]["content"][:600],
                "total_chars": len(fmt["messages"][2]["content"]),
            })

    report = {
        "teacher": TEACHER,
        "summary": {
            "total": len(formatted),
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "split": "80/10/10 stratified by source",
        },
        "stratified_split": split_report,
        "token_stats": {
            "full_assistant_turn": dist(total_tokens),
            "thinking_section": dist(thinking_tokens),
            "answer_section": dist(answer_tokens),
        },
        "format": {
            "system": SYSTEM_MESSAGE,
            "assistant_template": "<think>\n{thinking}\n</think>\n\n<answer>\n{response}\n</answer>",
        },
        "samples": samples,
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FORMAT RESULTS — {TEACHER.upper()}")
    print(f"{'='*60}")
    print(f"Total:      {len(formatted)}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")
    print()
    print("Stratified split by source:")
    print(f"  {'Source':12s} {'Total':>6s} {'Train':>6s} {'Val':>6s} {'Test':>6s}")
    for source, counts in split_report.items():
        print(f"  {source:12s} {counts['total']:6d} {counts['train']:6d} {counts['val']:6d} {counts['test']:6d}")
    print()
    if total_tokens:
        d = dist(total_tokens)
        print(f"Token length (full assistant turn):")
        print(f"  min={d['min']}  median={d['median']}  max={d['max']}  mean={d['mean']}")
    if thinking_tokens:
        d = dist(thinking_tokens)
        print(f"Token length (thinking section only):")
        print(f"  min={d['min']}  median={d['median']}  max={d['max']}  mean={d['mean']}")
    print()
    print("Sample per source:")
    for s in samples:
        print(f"  [{s['source']}] {s['problem'][:80]}...")
    print(f"\nFull report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
