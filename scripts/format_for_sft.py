"""Format filtered traces into Unsloth SFT chat format with train/val/test split.

Writes detailed formatting report and examples to data/format_report.json for the devlog.
"""

import json
import random
from pathlib import Path

random.seed(42)

INPUT_FILE = Path("data/traces_filtered.jsonl")
TRAIN_FILE = Path("data/train.jsonl")
VAL_FILE = Path("data/validation.jsonl")
TEST_FILE = Path("data/test.jsonl")
REPORT_FILE = Path("data/format_report.json")

SYSTEM_MESSAGE = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10


def format_entry(entry):
    """Convert a trace entry to chat message format with <think>/<answer> tags."""
    thinking = entry.get("thinking", "").strip()
    response = entry.get("response", "").strip()
    assistant_content = f"<think>\n{thinking}\n</think>\n\n<answer>\n{response}\n</answer>"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def count_tokens_approx(text):
    """Rough token count (words / 0.75)."""
    return int(len(text.split()) / 0.75)


def main():
    entries = []
    with open(INPUT_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    formatted = [format_entry(e) for e in entries]

    # Shuffle and split
    random.shuffle(formatted)
    train_end = int(len(formatted) * TRAIN_RATIO)
    val_end = train_end + int(len(formatted) * VAL_RATIO)
    train = formatted[:train_end]
    val = formatted[train_end:val_end]
    test = formatted[val_end:]

    # Write splits
    for data, path in [(train, TRAIN_FILE), (val, VAL_FILE), (test, TEST_FILE)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    # Measure token lengths across the dataset
    total_tokens = []
    thinking_tokens = []
    answer_tokens = []
    for entry in formatted:
        assistant = entry["messages"][2]["content"]
        total_tokens.append(count_tokens_approx(assistant))
        think_match = assistant[len("<think>\n"):assistant.find("\n</think>")]
        answer_match = assistant[assistant.find("<answer>\n") + len("<answer>\n"):assistant.find("\n</answer>")]
        thinking_tokens.append(count_tokens_approx(think_match))
        answer_tokens.append(count_tokens_approx(answer_match))

    def dist(vals):
        s = sorted(vals)
        n = len(s)
        return {
            "min": s[0], "p25": s[n//4], "median": s[n//2],
            "p75": s[3*n//4], "max": s[-1], "mean": int(sum(s)/n)
        }

    # Grab 2 full examples for devlog
    sample_entries = [entries[i] for i in range(min(2, len(entries)))]
    samples = []
    for e in sample_entries:
        fmt = format_entry(e)
        samples.append({
            "source": e["source"],
            "problem": e["problem"][:300],
            "expected_answer": e.get("expected_answer", "")[:100],
            "formatted_assistant_preview": fmt["messages"][2]["content"][:600],
            "total_chars": len(fmt["messages"][2]["content"]),
        })

    report = {
        "summary": {
            "total": len(formatted),
            "train": len(train),
            "validation": len(val),
            "test": len(test),
            "split": "80/10/10",
        },
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

    print(f"\n{'='*50}")
    print(f"FORMAT RESULTS")
    print(f"{'='*50}")
    print(f"Total:      {len(formatted)}")
    print(f"Train:      {len(train)}")
    print(f"Validation: {len(val)}")
    print(f"Test:       {len(test)}")
    print()
    print(f"Token length (full assistant turn):")
    d = report["token_stats"]["full_assistant_turn"]
    print(f"  min={d['min']}  median={d['median']}  max={d['max']}  mean={d['mean']}")
    print(f"Token length (thinking section only):")
    d = report["token_stats"]["thinking_section"]
    print(f"  min={d['min']}  median={d['median']}  max={d['max']}  mean={d['mean']}")
    print()
    print("Sample formatted entry:")
    print(f"  Source: {samples[0]['source']}")
    print(f"  Problem: {samples[0]['problem'][:100]}...")
    print(f"  Assistant preview:\n{samples[0]['formatted_assistant_preview'][:400]}")
    print(f"\nFull report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
