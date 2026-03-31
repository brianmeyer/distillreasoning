"""Generate reasoning traces from Kimi K2.5 cloud via Ollama.

Same pipeline as GLM-5 but using Kimi as the teacher model.
Runs with fewer workers to avoid rate limiting alongside GLM-5.
"""

import json
import time
import random
import threading
import ollama
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

random.seed(42)

INPUT_FILE = Path("data/problems.jsonl")
OUTPUT_FILE = Path("data/traces_raw_kimi.jsonl")

SYSTEM_PROMPT = (
    "You are a reasoning expert. Think through each problem step by step "
    "in detail before giving your final answer. Show all your work."
)

MODEL = "kimi-k2.5:cloud"
NUM_WORKERS = 4      # Match GLM-5 workers
DELAY_SECONDS = 0.5  # Match GLM-5 delay
MAX_RETRIES = 3
RETRY_BACKOFF = 5

write_lock = threading.Lock()
counter_lock = threading.Lock()
completed_count = 0


def load_problems():
    problems = []
    with open(INPUT_FILE) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def load_completed_ids():
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


def generate_trace(problem):
    global completed_count

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": problem["problem"]},
                ],
                think=True,
            )
            thinking = ""
            if hasattr(response.message, "thinking") and response.message.thinking:
                thinking = response.message.thinking
            content = response.message.content or ""

            entry = {
                "id": problem["id"],
                "source": problem["source"],
                "problem": problem["problem"],
                "expected_answer": problem.get("expected_answer", ""),
                "thinking": thinking,
                "response": content,
                "teacher": "kimi-k2.5",
            }

            with write_lock:
                with open(OUTPUT_FILE, "a") as f:
                    f.write(json.dumps(entry) + "\n")
                    f.flush()

            think_len = len(thinking.split()) if thinking else 0

            with counter_lock:
                completed_count += 1
                current = completed_count

            return problem["id"], problem["source"], think_len, current

        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF * attempt
                time.sleep(wait)
            else:
                with counter_lock:
                    completed_count += 1
                    current = completed_count
                return problem["id"], problem["source"], -1, current


def main():
    global completed_count

    problems = load_problems()
    completed_ids = load_completed_ids()
    remaining = [p for p in problems if p["id"] not in completed_ids]
    total = len(problems)
    done = len(completed_ids)
    completed_count = done

    print(f"Model: {MODEL}")
    print(f"Total problems: {total}")
    print(f"Already completed: {done}")
    print(f"Remaining: {len(remaining)}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        for i, problem in enumerate(remaining):
            future = executor.submit(generate_trace, problem)
            futures[future] = problem
            if i < NUM_WORKERS:
                time.sleep(0.3)
            else:
                time.sleep(DELAY_SECONDS)

        for future in as_completed(futures):
            pid, source, think_len, current = future.result()
            elapsed = time.time() - start_time
            rate = (current - done) / elapsed * 3600 if elapsed > 0 else 0

            if think_len >= 0:
                print(f"[{current}/{total}] {pid} ({source}) — {think_len} words [{rate:.0f}/hr]")
            else:
                print(f"[{current}/{total}] {pid} ({source}) — FAILED")

    final_count = sum(1 for _ in open(OUTPUT_FILE))
    elapsed = time.time() - start_time
    print(f"\nDone! {final_count} Kimi traces in {elapsed/3600:.1f} hours")


if __name__ == "__main__":
    main()
