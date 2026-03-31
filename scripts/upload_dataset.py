"""Upload datasets to HuggingFace Hub.

Publishes two datasets:
  1. bmeyer2025/glm5-reasoning-traces       — raw traces (problem, thinking, response)
  2. bmeyer2025/glm5-reasoning-traces-sft   — formatted train/val/test for SFT

Set HF_TOKEN environment variable before running.
"""

import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login, HfApi

HF_USERNAME = "bmeyer2025"
RAW_REPO = f"{HF_USERNAME}/glm5-reasoning-traces"
SFT_REPO = f"{HF_USERNAME}/glm5-reasoning-traces-sft"

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    login()


# ── 1. Raw traces ──────────────────────────────────────────────────────────────
print("Loading raw filtered traces...")
raw_ds = load_dataset("json", data_files={"train": "data/traces_filtered.jsonl"})

print(f"Raw traces: {len(raw_ds['train'])} examples")
print(f"Columns: {raw_ds['train'].column_names}")
# Each row: id, source, problem, expected_answer, thinking, response

print(f"Pushing raw traces to {RAW_REPO}...")
raw_ds.push_to_hub(
    RAW_REPO,
    commit_message="Add GLM-5 reasoning traces (problem, thinking, response)",
)
print(f"  ✅ https://huggingface.co/datasets/{RAW_REPO}")


# ── 2. SFT-formatted splits ────────────────────────────────────────────────────
print("\nLoading SFT-formatted splits...")
sft_ds = load_dataset("json", data_files={
    "train":      "data/train.jsonl",
    "validation": "data/validation.jsonl",
    "test":       "data/test.jsonl",
})

print(f"Train:      {len(sft_ds['train'])} examples")
print(f"Validation: {len(sft_ds['validation'])} examples")
print(f"Test:       {len(sft_ds['test'])} examples")
# Each row: messages (list of system/user/assistant dicts with <think>/<answer> tags)

print(f"Pushing SFT dataset to {SFT_REPO}...")
sft_ds.push_to_hub(
    SFT_REPO,
    commit_message="Add SFT-formatted reasoning traces (train/val/test, 80/10/10)",
)
print(f"  ✅ https://huggingface.co/datasets/{SFT_REPO}")

# ── 3. Upload dataset cards ────────────────────────────────────────────────────
print("\nUploading dataset card...")
api = HfApi()
api.upload_file(
    path_or_fileobj="cards/dataset_card.md",
    path_in_repo="README.md",
    repo_id=RAW_REPO,
    repo_type="dataset",
    commit_message="Add dataset card",
)
print(f"  ✅ Dataset card uploaded to {RAW_REPO}")

print("\nDone!")
print(f"  Raw traces:    https://huggingface.co/datasets/{RAW_REPO}")
print(f"  SFT-formatted: https://huggingface.co/datasets/{SFT_REPO}")
