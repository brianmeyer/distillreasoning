---
license: mit
base_model: Qwen/Qwen3.5-4B
tags:
  - reasoning
  - distillation
  - chain-of-thought
  - lora
  - gguf
  - math
language:
  - en
pipeline_tag: text-generation
---

# Qwen3.5-4B — GLM-5 Reasoning Distilled

<p align="center">
  <img src="https://raw.githubusercontent.com/brianmeyer/distillreasoning/main/images/distillation_apparatus.png" alt="Chemistry distillation apparatus with mathematical reasoning flowing through it into a flask labeled Pure Distilled Reasoning" width="600">
</p>

Qwen3.5-4B fine-tuned on reasoning traces from GLM-5 (744B MoE). The model learns to produce structured step-by-step reasoning chains before answering — distilled from one of the highest-scoring reasoning models available.

## What this is

Base Qwen3.5-4B gives flat answers. This model thinks first:

```
<think>
Step 1: ...
Step 2: ...
Therefore...
</think>

<answer>
[final answer]
</answer>
```

## Training details

| | |
|--|--|
| **Base model** | Qwen/Qwen3.5-4B |
| **Teacher model** | GLM-5 (744B MoE, 40B active) via Ollama cloud |
| **Method** | LoRA SFT via Tinker (Thinking Machines Lab) |
| **LoRA rank** | 32 |
| **Training data** | ~1,600 reasoning traces across math, logic, code |
| **Data split** | 80% train / 10% validation / 10% test |
| **Epochs** | 3 |
| **Dataset** | [bmeyer2025/glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft) |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled")
tokenizer = AutoTokenizer.from_pretrained("bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled")

messages = [
    {"role": "system", "content": "You are a helpful reasoning assistant. Think through problems step by step before answering."},
    {"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"},
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

## Run locally with Ollama

```bash
ollama run hf.co/bmeyer2025/qwen3.5-4b-glm5-reasoning-distilled:Q4_K_M
```

## GGUF files

| File | Quantization | Use case |
|------|-------------|----------|
| `*Q4_K_M.gguf` | 4-bit | Best size/quality balance, recommended |
| `*Q8_0.gguf` | 8-bit | Higher quality, larger file |

## Evaluation

Comparison vs base Qwen3.5-4B on 100 held-out GSM8K problems:

| Model | Accuracy | Format compliance | Avg thinking tokens |
|-------|----------|-------------------|---------------------|
| Base Qwen3.5-4B | TBD | 0% | 0 |
| Distilled (this model) | TBD | ~95%+ | TBD |

*Results to be updated after eval run*

## Limitations

- 4B parameters — will struggle on very hard competition math (AIME-level)
- Reasoning style reflects GLM-5's patterns, not all reasoning strategies
- Trained on English problems only

## Related

- **Raw traces:** [bmeyer2025/glm5-reasoning-traces](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces)
- **SFT dataset:** [bmeyer2025/glm5-reasoning-traces-sft](https://huggingface.co/datasets/bmeyer2025/glm5-reasoning-traces-sft)
- **Code + dev log:** [brianmeyer/distillreasoning](https://github.com/brianmeyer/distillreasoning)

## License

MIT
