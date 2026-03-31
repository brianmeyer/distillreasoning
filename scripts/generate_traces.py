"""Generate reasoning traces from GLM-5 cloud via Ollama for each problem."""

import json
import time
import random
import ollama
from pathlib import Path

random.seed(42)

INPUT_FILE = Path("data/problems.jsonl")
OUTPUT_FILE = Path("data/traces_raw.jsonl")

SYSTEM_PROMPT = (
    "You are a reasoning expert. Think through each problem step by step "
    "in detail before giving your final answer. Show all your work."
)

MODEL = "glm-5:cloud"
MAX_PROBLEMS = 1200  # Target count from the guide
DELAY_SECONDS = 2    # Rate limiting
MAX_RETRIES = 3
RETRY_BACKOFF = 5    # seconds, multiplied by attempt number


def load_problems():
    problems = []
    with open(INPUT_FILE) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def load_completed_ids():
    """Load IDs already completed so we can resume."""
    completed = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    completed.add(entry["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def generate_trace(problem_text):
    """Send a problem to GLM-5 and capture the full response with thinking."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem_text},
                ],
                think=True,
            )
            thinking = ""
            if hasattr(response.message, "thinking") and response.message.thinking:
                thinking = response.message.thinking
            content = response.message.content or ""
            return thinking, content
        except Exception as e:
            print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    return None, None


def main():
    problems = load_problems()
    completed_ids = load_completed_ids()

    # Sample down to MAX_PROBLEMS if we have more
    if len(problems) > MAX_PROBLEMS:
        # Stratified sample: keep proportional representation from each source
        by_source = {}
        for p in problems:
            by_source.setdefault(p["source"], []).append(p)

        sampled = []
        for source, items in by_source.items():
            ratio = len(items) / len(problems)
            n = max(1, int(ratio * MAX_PROBLEMS))
            sampled.extend(random.sample(items, min(n, len(items))))
        problems = sampled[:MAX_PROBLEMS]
        random.shuffle(problems)

    # Filter out already completed
    remaining = [p for p in problems if p["id"] not in completed_ids]

    total = len(problems)
    done = len(completed_ids)
    print(f"Total problems: {total}")
    print(f"Already completed: {done}")
    print(f"Remaining: {len(remaining)}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with open(OUTPUT_FILE, "a") as f:
        for i, problem in enumerate(remaining):
            idx = done + i + 1
            print(f"[{idx}/{total}] {problem['id']} ({problem['source']})...", end=" ", flush=True)

            thinking, content = generate_trace(problem["problem"])

            if thinking is None and content is None:
                print("FAILED (all retries exhausted)")
                continue

            entry = {
                "id": problem["id"],
                "source": problem["source"],
                "problem": problem["problem"],
                "expected_answer": problem.get("expected_answer", ""),
                "thinking": thinking,
                "response": content,
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()

            think_len = len(thinking.split()) if thinking else 0
            print(f"OK (thinking: {think_len} words)")

            time.sleep(DELAY_SECONDS)

    # Final stats
    final_count = sum(1 for _ in open(OUTPUT_FILE))
    print(f"\nDone! {final_count} traces saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
