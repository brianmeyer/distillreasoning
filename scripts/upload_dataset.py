"""Upload all datasets to HuggingFace Hub.

Publishes:
  1. bmeyer2025/glm5-reasoning-traces       — raw GLM-5 traces
  2. bmeyer2025/glm5-reasoning-traces-sft   — formatted GLM-5 train/val/test
  3. bmeyer2025/kimi-reasoning-traces        — raw Kimi traces
  4. bmeyer2025/kimi-reasoning-traces-sft    — formatted Kimi train/val/test

Set HF_TOKEN environment variable before running.
"""

import os
from datasets import load_dataset
from huggingface_hub import login, HfApi

HF_USERNAME = "bmeyer2025"

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
else:
    login()

api = HfApi()

datasets_to_push = [
    {
        "repo": f"{HF_USERNAME}/glm5-reasoning-traces",
        "files": {"train": "data/traces_filtered_glm5.jsonl"},
        "card": "cards/dataset_card.md",
        "msg": "Add GLM-5 reasoning traces",
    },
    {
        "repo": f"{HF_USERNAME}/glm5-reasoning-traces-sft",
        "files": {
            "train": "data/train_glm5.jsonl",
            "validation": "data/validation_glm5.jsonl",
        },
        "card": None,
        "msg": "Add GLM-5 SFT-formatted traces (train/val)",
    },
    {
        "repo": f"{HF_USERNAME}/kimi-reasoning-traces",
        "files": {"train": "data/traces_filtered_kimi.jsonl"},
        "card": None,
        "msg": "Add Kimi K2.5 reasoning traces",
    },
    {
        "repo": f"{HF_USERNAME}/kimi-reasoning-traces-sft",
        "files": {
            "train": "data/train_kimi.jsonl",
            "validation": "data/validation_kimi.jsonl",
        },
        "card": None,
        "msg": "Add Kimi K2.5 SFT-formatted traces (train/val)",
    },
]

for ds_config in datasets_to_push:
    repo = ds_config["repo"]
    print(f"\n{'='*50}")
    print(f"Pushing: {repo}")
    print(f"{'='*50}")

    ds = load_dataset("json", data_files=ds_config["files"])
    for split_name, split_data in ds.items():
        print(f"  {split_name}: {len(split_data)} examples")

    ds.push_to_hub(repo, commit_message=ds_config["msg"])
    print(f"  ✅ https://huggingface.co/datasets/{repo}")

    if ds_config["card"]:
        api.upload_file(
            path_or_fileobj=ds_config["card"],
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="Add dataset card",
        )
        print(f"  ✅ Dataset card uploaded")

print(f"\n{'='*50}")
print("All datasets pushed!")
for ds_config in datasets_to_push:
    print(f"  https://huggingface.co/datasets/{ds_config['repo']}")
