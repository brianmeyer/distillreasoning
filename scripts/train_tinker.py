"""SFT training on Tinker (Thinking Machines Lab).

Trains Qwen3.5-4B with LoRA on GLM-5 reasoning traces.
Set TINKER_API_KEY environment variable before running.

Usage:
    export TINKER_API_KEY=your_key
    python scripts/train_tinker.py
"""

import json
import random
import numpy as np
from pathlib import Path

import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lr

# Config
MODEL_NAME = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5"
LORA_RANK = 32
BATCH_SIZE = 128
NUM_EPOCHS = 3
EVAL_EVERY = 50  # Evaluate every N steps
SAVE_EVERY = 200  # Checkpoint every N steps
# Local files (used if HuggingFace dataset not yet uploaded)
TRAIN_FILE = Path("data/train.jsonl")
VAL_FILE = Path("data/validation.jsonl")
# Once uploaded, can also load from HF:
# from datasets import load_dataset
# ds = load_dataset("bmeyer2025/glm5-reasoning-traces-sft")
# train_convos = [ex["messages"] for ex in ds["train"]]
LOG_FILE = Path("data/tinker_metrics.jsonl")

random.seed(42)


def load_conversations(path):
    """Load JSONL chat data."""
    convos = []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            convos.append(entry["messages"])
    return convos


def prepare_datum(messages, renderer):
    """Convert a conversation to a Tinker Datum with proper weight masking."""
    model_input, weights = renderer.build_supervised_example(messages)

    # Build target tokens (shifted by 1)
    tokens = list(model_input.tokens)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = list(weights[1:])  # Shift weights to match targets

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(
            weights=weights,
            target_tokens=target_tokens,
        ),
    )


def compute_loss(fwdbwd_result, data_batch):
    """Compute average loss from forward-backward results."""
    logprobs = np.concatenate(
        [output["logprobs"].tolist() for output in fwdbwd_result.loss_fn_outputs]
    )
    weights = np.concatenate(
        [example.loss_fn_inputs["weights"] for example in data_batch]
    )
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return 0.0
    return float(-np.dot(logprobs, weights) / weight_sum)


def main():
    print(f"=== Tinker SFT Training ===")
    print(f"Model: {MODEL_NAME}")
    print(f"LoRA rank: {LORA_RANK}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()

    # Load data
    print("Loading training data...")
    train_convos = load_conversations(TRAIN_FILE)
    val_convos = load_conversations(VAL_FILE)
    print(f"  Train: {len(train_convos)} conversations")
    print(f"  Validation: {len(val_convos)} conversations")

    # Initialize Tinker
    print("Connecting to Tinker...")
    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=MODEL_NAME,
        rank=LORA_RANK,
    )
    tokenizer = tokenizer_utils.get_tokenizer(MODEL_NAME)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer)

    # Get recommended learning rate
    lr = get_lr(MODEL_NAME)
    print(f"  Recommended LR: {lr}")

    # Prepare all data as Datum objects
    print("Preparing training data...")
    train_data = []
    skipped = 0
    for messages in train_convos:
        try:
            datum = prepare_datum(messages, renderer)
            train_data.append(datum)
        except Exception as e:
            skipped += 1
    print(f"  Prepared: {len(train_data)} examples ({skipped} skipped)")

    print("Preparing validation data...")
    val_data = []
    for messages in val_convos[:50]:  # Use subset for faster eval
        try:
            datum = prepare_datum(messages, renderer)
            val_data.append(datum)
        except Exception:
            pass
    print(f"  Prepared: {len(val_data)} validation examples")

    # Training loop
    total_steps = (len(train_data) * NUM_EPOCHS) // BATCH_SIZE
    print(f"\nStarting training: {total_steps} total steps")
    print(f"{'='*60}")

    step = 0
    metrics_log = []

    for epoch in range(NUM_EPOCHS):
        random.shuffle(train_data)

        for batch_start in range(0, len(train_data), BATCH_SIZE):
            batch = train_data[batch_start : batch_start + BATCH_SIZE]
            if not batch:
                continue

            # Forward-backward pass
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")

            # Optimizer step
            optim_future = training_client.optim_step(
                types.AdamParams(learning_rate=lr)
            )

            # Get results
            fwdbwd_result = fwdbwd_future.result()
            optim_result = optim_future.result()

            train_loss = compute_loss(fwdbwd_result, batch)
            step += 1

            metric = {"step": step, "epoch": epoch + 1, "train_loss": train_loss}

            # Evaluation
            if step % EVAL_EVERY == 0 or step == 1:
                val_fwd = training_client.forward_backward(
                    val_data, "cross_entropy"
                ).result()
                val_loss = compute_loss(val_fwd, val_data)
                metric["val_loss"] = val_loss
                print(
                    f"[Step {step}/{total_steps}] "
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Train loss: {train_loss:.4f} | "
                    f"Val loss: {val_loss:.4f}"
                )
            else:
                print(
                    f"[Step {step}/{total_steps}] "
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Train loss: {train_loss:.4f}"
                )

            metrics_log.append(metric)

            # Save checkpoint
            if step % SAVE_EVERY == 0:
                print(f"  Saving checkpoint at step {step}...")
                save_result = training_client.save_state(
                    name=f"distill-reasoning-step-{step}"
                ).result()
                print(f"  Saved: {save_result.path}")

        print(f"--- Epoch {epoch+1} complete ---")

    # Final save
    print("\nSaving final model...")
    final_save = training_client.save_state(
        name="distill-reasoning-final"
    ).result()
    print(f"Final checkpoint: {final_save.path}")

    # Save for sampling/inference
    print("Creating sampling checkpoint...")
    sampling_client = training_client.save_weights_and_get_sampling_client(
        name="distill-reasoning-final-sampler"
    )
    print("Sampling client ready!")

    # Test generation
    print("\n=== Test Generation ===")
    test_problems = [
        "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    ]

    for problem in test_problems:
        messages = [
            {"role": "system", "content": "You are a helpful reasoning assistant. Think through problems step by step before answering."},
            {"role": "user", "content": problem},
        ]
        prompt = renderer.build_generation_prompt(messages)
        stop_sequences = renderer.get_stop_sequences()

        result = sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(
                max_tokens=1024,
                temperature=0.7,
                stop=stop_sequences,
            ),
            num_samples=1,
        ).result()

        response_text = tokenizer.decode(result.sequences[0].tokens)
        print(f"\nProblem: {problem}")
        print(f"Response: {response_text[:500]}")

    # Save metrics
    with open(LOG_FILE, "w") as f:
        for m in metrics_log:
            f.write(json.dumps(m) + "\n")
    print(f"\nMetrics saved to {LOG_FILE}")
    print("Done!")


if __name__ == "__main__":
    main()
