"""Format filtered traces into Unsloth SFT chat format with train/test split."""

import json
import random
from pathlib import Path

random.seed(42)

INPUT_FILE = Path("data/traces_filtered.jsonl")
TRAIN_FILE = Path("data/train.jsonl")
TEST_FILE = Path("data/test.jsonl")

SYSTEM_MESSAGE = "You are a helpful reasoning assistant. Think through problems step by step before answering."

TRAIN_RATIO = 0.9


def format_entry(entry):
    """Convert a trace entry to chat message format with <think>/<answer> tags."""
    thinking = entry.get("thinking", "").strip()
    response = entry.get("response", "").strip()

    # Build assistant content with think/answer tags
    assistant_content = f"<think>\n{thinking}\n</think>\n\n<answer>\n{response}\n</answer>"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": entry["problem"]},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    # Load filtered traces
    entries = []
    with open(INPUT_FILE) as f:
        for line in f:
            entries.append(json.loads(line))

    # Format all
    formatted = [format_entry(e) for e in entries]

    # Shuffle and split
    random.shuffle(formatted)
    split_idx = int(len(formatted) * TRAIN_RATIO)
    train = formatted[:split_idx]
    test = formatted[split_idx:]

    # Write
    for data, path in [(train, TRAIN_FILE), (test, TEST_FILE)]:
        with open(path, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")

    print(f"Formatted {len(formatted)} entries")
    print(f"  Train: {len(train)} -> {TRAIN_FILE}")
    print(f"  Test:  {len(test)} -> {TEST_FILE}")

    # Print a sample
    print("\n--- Sample formatted entry ---")
    sample = formatted[0]
    print(f"System: {sample['messages'][0]['content'][:80]}...")
    print(f"User: {sample['messages'][1]['content'][:100]}...")
    print(f"Assistant: {sample['messages'][2]['content'][:200]}...")


if __name__ == "__main__":
    main()
