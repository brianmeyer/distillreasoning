"""Train 6 distilled models on Tinker: 2 students × 3 teacher configs.

Usage: TINKER_API_KEY=xxx python scripts/train_tinker.py
"""

import tinker
from tinker_cookbook.supervised import conversation_to_datum, compute_mean_nll
from tinker_cookbook.renderers import TrainOnWhat, get_renderer
from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.hyperparam_utils import get_lr
import json, time, os, sys
from pathlib import Path

BATCH_SIZE = 8
MAX_LENGTH = 4096
EVAL_EVERY = 50
SAVE_EVERY = 200
NUM_EPOCHS = 3

RUNS = [
    {"student": "Qwen/Qwen3.5-4B",  "teacher": "glm5",     "data": "data/train_glm5.jsonl",     "val": "data/validation_glm5.jsonl"},
    {"student": "Qwen/Qwen3.5-4B",  "teacher": "kimi",     "data": "data/train_kimi.jsonl",      "val": "data/validation_kimi.jsonl"},
    {"student": "Qwen/Qwen3.5-4B",  "teacher": "combined", "data": "data/train_combined.jsonl",   "val": "data/validation_combined.jsonl"},
]

# Allow starting from a specific run index
START_FROM = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def run_training(run_config, run_idx):
    student = run_config["student"]
    teacher = run_config["teacher"]
    student_short = student.split("/")[-1].lower().replace(".", "")

    run_name = f"{student_short}-{teacher}"
    print(f"\n{'='*60}")
    print(f"RUN {run_idx+1}/6: {run_name}")
    print(f"  Student: {student}")
    print(f"  Teacher: {teacher}")
    print(f"  Data: {run_config['data']}")
    print(f"{'='*60}")

    # Load data
    train_data = load_jsonl(run_config["data"])
    val_data = load_jsonl(run_config["val"])
    print(f"  Train: {len(train_data)} examples, Val: {len(val_data)} examples")

    # Setup Tinker
    print(f"  Creating training client...")
    service = tinker.ServiceClient()
    client = service.create_lora_training_client(
        base_model=student, rank=32, train_mlp=True, train_attn=True,
    )
    tokenizer = tokenizer_utils.get_tokenizer(student)
    renderer = get_renderer(model_info.get_recommended_renderer_name(student), tokenizer)
    lr = get_lr(student)
    print(f"  LR: {lr:.6f}")

    # Build all datums upfront
    print(f"  Building train datums...")
    train_datums = []
    for ex in train_data:
        try:
            d = conversation_to_datum(ex["messages"], renderer, MAX_LENGTH,
                                      train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            train_datums.append(d)
        except Exception as e:
            pass  # Skip malformed examples
    print(f"  Built {len(train_datums)} train datums")

    print(f"  Building val datums...")
    val_datums = []
    for ex in val_data:
        try:
            d = conversation_to_datum(ex["messages"], renderer, MAX_LENGTH,
                                      train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE)
            val_datums.append(d)
        except Exception:
            pass
    print(f"  Built {len(val_datums)} val datums")

    # Training loop
    import random
    random.seed(42)

    n_batches_per_epoch = len(train_datums) // BATCH_SIZE
    total_steps = n_batches_per_epoch * NUM_EPOCHS
    print(f"  Steps per epoch: {n_batches_per_epoch}, Total: {total_steps}")
    print(f"  Training...")

    log_file = Path(f"data/train_log_{run_name}.jsonl")
    best_val_loss = float("inf")
    step = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        random.shuffle(train_datums)

        for batch_start in range(0, len(train_datums) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = train_datums[batch_start:batch_start + BATCH_SIZE]
            step += 1

            # Linear LR decay
            lr_mult = max(0.0, 1.0 - step / total_steps)
            current_lr = lr * lr_mult
            adam = tinker.AdamParams(learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8)

            fwd = client.forward_backward(batch, "cross_entropy")
            opt = client.optim_step(adam)

            fwd_result = fwd.result()
            opt.result()

            logprobs = [x["logprobs"] for x in fwd_result.loss_fn_outputs]
            weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(logprobs, weights)

            # Log
            entry = {"step": step, "epoch": epoch + 1, "train_nll": round(train_nll, 4),
                     "lr": round(current_lr, 8)}

            # Eval
            if step % EVAL_EVERY == 0 or step == 1:
                val_batch = val_datums[:min(BATCH_SIZE * 4, len(val_datums))]
                val_fwd = client.forward_backward(val_batch, "cross_entropy")
                val_result = val_fwd.result()
                val_logprobs = [x["logprobs"] for x in val_result.loss_fn_outputs]
                val_weights = [d.loss_fn_inputs["weights"] for d in val_batch]
                val_nll = compute_mean_nll(val_logprobs, val_weights)
                entry["val_nll"] = round(val_nll, 4)

                if val_nll < best_val_loss:
                    best_val_loss = val_nll
                    entry["best"] = True

                elapsed = time.time() - start_time
                print(f"  [{step}/{total_steps}] train={train_nll:.4f} val={val_nll:.4f} lr={current_lr:.6f} ({elapsed/60:.1f}m)")

            elif step % 10 == 0:
                elapsed = time.time() - start_time
                print(f"  [{step}/{total_steps}] train={train_nll:.4f} lr={current_lr:.6f} ({elapsed/60:.1f}m)")

            with open(log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

            # Save checkpoint
            if step % SAVE_EVERY == 0:
                save_name = f"{run_name}-step{step}"
                print(f"  Saving checkpoint: {save_name}")
                client.save_weights_for_sampler(name=save_name).result()

    # Final save
    final_name = f"{run_name}-final"
    print(f"  Saving final: {final_name}")
    save_result = client.save_weights_for_sampler(name=final_name).result()

    elapsed = time.time() - start_time
    print(f"\n  ✅ {run_name} complete! {step} steps in {elapsed/60:.1f} min")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final checkpoint: {final_name}")
    print(f"  Log: {log_file}")

    return {"run_name": run_name, "steps": step, "best_val_loss": best_val_loss,
            "time_min": round(elapsed / 60, 1), "final_checkpoint": final_name}


if __name__ == "__main__":
    results = []
    for i, run_config in enumerate(RUNS):
        if i < START_FROM:
            print(f"Skipping run {i+1} ({run_config['teacher']})")
            continue
        result = run_training(run_config, i)
        results.append(result)

    print(f"\n{'='*60}")
    print("ALL RUNS COMPLETE")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r['run_name']:30s} val={r['best_val_loss']:.4f} ({r['time_min']}min)")
