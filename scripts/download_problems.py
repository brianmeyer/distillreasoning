"""Download and combine reasoning problems from multiple HuggingFace datasets."""

import json
import random
from pathlib import Path
from datasets import load_dataset

random.seed(42)
OUTPUT_FILE = Path("data/problems.jsonl")
OUTPUT_FILE.parent.mkdir(exist_ok=True)

problems = []

# 1. GSM8K test set (~1,319 math word problems)
print("Downloading GSM8K test set...")
gsm8k = load_dataset("openai/gsm8k", "main", split="test")
for i, row in enumerate(gsm8k):
    # GSM8K answer format: "#### <number>" at the end
    answer_text = row["answer"]
    final_answer = answer_text.split("####")[-1].strip() if "####" in answer_text else ""
    problems.append({
        "id": f"gsm8k_{i}",
        "source": "gsm8k",
        "problem": row["question"],
        "expected_answer": final_answer,
    })
print(f"  GSM8K: {len([p for p in problems if p['source'] == 'gsm8k'])} problems")

# 2. MATH dataset - sample 200 competition-level problems
print("Downloading MATH dataset...")
math_ds = load_dataset("SuperSecureHuman/competition_math_hf_dataset", split="test")
math_indices = random.sample(range(len(math_ds)), min(200, len(math_ds)))
for i in math_indices:
    row = math_ds[i]
    # MATH answers are in \boxed{} format within the solution
    answer = row.get("solution", "")
    problems.append({
        "id": f"math_{i}",
        "source": "math",
        "problem": row["problem"],
        "expected_answer": answer,
    })
print(f"  MATH: {len([p for p in problems if p['source'] == 'math'])} problems")

# 3. ARC-Challenge test set (science reasoning)
print("Downloading ARC-Challenge test set...")
arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
arc_indices = random.sample(range(len(arc)), min(400, len(arc)))
for i in arc_indices:
    row = arc[i]
    choices = row["choices"]
    # Format as multiple choice
    choice_text = "\n".join(
        f"{label}. {text}"
        for label, text in zip(choices["label"], choices["text"])
    )
    problem = f"{row['question']}\n\nChoices:\n{choice_text}"
    problems.append({
        "id": f"arc_{i}",
        "source": "arc",
        "problem": problem,
        "expected_answer": row["answerKey"],
    })
print(f"  ARC: {len([p for p in problems if p['source'] == 'arc'])} problems")

# 4. HumanEval coding problems
print("Downloading HumanEval dataset...")
humaneval = load_dataset("openai/openai_humaneval", split="test")
for i, row in enumerate(humaneval):
    problems.append({
        "id": f"humaneval_{i}",
        "source": "humaneval",
        "problem": f"Write a Python function to solve the following:\n\n{row['prompt']}",
        "expected_answer": row.get("canonical_solution", ""),
    })
print(f"  HumanEval: {len([p for p in problems if p['source'] == 'humaneval'])} problems")

# Shuffle and write
random.shuffle(problems)
with open(OUTPUT_FILE, "w") as f:
    for p in problems:
        f.write(json.dumps(p) + "\n")

print(f"\nTotal: {len(problems)} problems written to {OUTPUT_FILE}")
print("Breakdown:")
for source in ["gsm8k", "math", "arc", "humaneval"]:
    count = len([p for p in problems if p["source"] == source])
    print(f"  {source}: {count}")
