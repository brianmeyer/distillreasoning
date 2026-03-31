# GLM-5 Reasoning Distillation into Qwen3.5-4B

Distill reasoning capabilities from GLM-5 (744B MoE, 40B active params) into Qwen3.5-4B using Unsloth QLoRA SFT + optional GRPO reinforcement learning.

## Deliverables

- **Dataset:** [bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces) - ~800-1,000 reasoning traces across math, logic, and code
- **Model:** [bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled](https://huggingface.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled) - Fine-tuned model with GGUF exports
- **Cost:** $0 (Ollama cloud for generation, Colab free tier for training)

## Pipeline

1. **Problem Collection** - GSM8K, MATH, ARC-Challenge, HumanEval (~1,200 problems)
2. **Trace Generation** - GLM-5 via Ollama cloud with full `<think>` reasoning traces
3. **Filtering** - Verify correctness, drop short/repetitive traces (~800-1,000 kept)
4. **SFT Training** - Qwen3.5-4B with QLoRA via Unsloth on Colab T4
5. **GRPO (Optional)** - Reinforce correct reasoning with reward functions
6. **Evaluation** - Compare base vs SFT vs GRPO on held-out problems
7. **Export** - GGUF for local inference via Ollama/llama.cpp

## Project Structure

```
distillreasoning/
├── scripts/
│   ├── download_problems.py   # Download problem sets from HuggingFace
│   ├── generate_traces.py     # Generate reasoning traces via GLM-5
│   ├── filter_traces.py       # Filter for quality and correctness
│   ├── format_for_sft.py      # Format into Unsloth SFT chat format
│   ├── upload_dataset.py      # Push dataset to HuggingFace
│   └── evaluate.py            # Evaluation script (run in Colab)
├── notebooks/
│   └── sft_training.ipynb     # Colab notebook for SFT training
├── data/                      # Generated data (not committed)
│   ├── problems.jsonl
│   ├── traces_raw.jsonl
│   ├── traces_filtered.jsonl
│   ├── train.jsonl
│   └── test.jsonl
└── README.md
```

## Quick Start

### 1. Generate Dataset (Local)

```bash
python3 -m venv venv && source venv/bin/activate
pip install ollama datasets huggingface_hub

# Download problems
python scripts/download_problems.py

# Generate traces (takes several hours)
python scripts/generate_traces.py

# Filter and format
python scripts/filter_traces.py
python scripts/format_for_sft.py

# Upload to HuggingFace
HF_TOKEN=your_token python scripts/upload_dataset.py
```

### 2. Train (Google Colab)

Open `notebooks/sft_training.ipynb` in Google Colab with T4 GPU and run all cells.

### 3. Run Locally

```bash
# After GGUF is published to HuggingFace
ollama run hf.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled
```

## Teacher Model

GLM-5 was chosen as the teacher because:
- Highest math/reasoning benchmarks (92.7% AIME 2026, 86% GPQA-Diamond)
- Full `<think>` traces exposed
- MIT licensed, open weights
- Free via Ollama cloud tag

## License

MIT
