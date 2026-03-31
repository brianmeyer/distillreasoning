"""Upload the formatted dataset to HuggingFace Hub."""

from datasets import load_dataset
from huggingface_hub import login
import os

REPO_ID = "bmeyer2025/glm5-reasoning-traces"

# Login (set HF_TOKEN env var or it will prompt)
token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    login()

# Load train/test splits
dataset = load_dataset("json", data_files={
    "train": "data/train.jsonl",
    "test": "data/test.jsonl",
})

print(f"Train: {len(dataset['train'])} examples")
print(f"Test: {len(dataset['test'])} examples")

# Push to hub
dataset.push_to_hub(REPO_ID)
print(f"\nDataset pushed to https://huggingface.co/datasets/{REPO_ID}")
