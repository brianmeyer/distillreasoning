"""GRPO (Group Relative Policy Optimization) on top of SFT checkpoints.

Loads each SFT checkpoint and runs RL training on GSM8K problems.
The model generates multiple answers per problem, gets rewarded for
correct ones, and learns to prefer reasoning chains that lead to
right answers.

Usage: TINKER_API_KEY=xxx python scripts/train_grpo.py
"""

import asyncio
import sys
import os

import chz

from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train

MODEL = "Qwen/Qwen3.5-4B"

# SFT checkpoints to start GRPO from
SFT_CHECKPOINTS = {
    "glm5": "tinker://0fbca836-2aae-5500-b28d-93c2a46a328b:train:0/sampler_weights/qwen35-4b-glm5-final",
    "kimi": "tinker://f5795e66-71e4-5cf4-9ebe-1cc14c27aa6e:train:0/sampler_weights/qwen35-4b-kimi-final",
    "combined": "tinker://41e7bd9e-e49f-5f13-a5ff-f4339faab448:train:0/sampler_weights/qwen35-4b-combined-final",
}

# Allow selecting which checkpoint to train: python train_grpo.py [glm5|kimi|combined]
TEACHER = sys.argv[1] if len(sys.argv) > 1 else "kimi"


def build_config_blueprint() -> chz.Blueprint[train.Config]:
    renderer_name = model_info.get_recommended_renderer_name(MODEL)

    builder = Gsm8kDatasetBuilder(
        batch_size=64,
        group_size=8,       # 8 responses per problem
        renderer_name=renderer_name,
        model_name_for_tokenizer=MODEL,
    )

    checkpoint_path = SFT_CHECKPOINTS[TEACHER]
    log_path = f"/tmp/tinker-grpo/qwen35-4b-{TEACHER}"

    return chz.Blueprint(train.Config).apply(
        {
            "model_name": MODEL,
            "renderer_name": renderer_name,
            "log_path": log_path,
            "dataset_builder": builder,
            "learning_rate": 2e-5,      # Lower LR for RL (vs 5e-4 for SFT)
            "max_tokens": 1024,         # Enough for reasoning chains
            "lora_rank": 32,
            "load_checkpoint_path": checkpoint_path,  # Start from SFT
            "eval_every": 20,
            "save_every": 20,
            "temperature": 0.8,         # Encourage exploration
        }
    )


def main(config: train.Config):
    print(f"GRPO Training: {TEACHER}")
    print(f"  Model: {MODEL}")
    print(f"  SFT checkpoint: {SFT_CHECKPOINTS[TEACHER]}")
    print(f"  Log path: {config.log_path}")
    print(f"  Group size: 8 (responses per problem)")
    print(f"  Learning rate: {config.learning_rate}")
    print()

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[2:])  # Skip the teacher arg
    main(blueprint.make())
