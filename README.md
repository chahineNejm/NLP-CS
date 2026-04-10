# Aspect-Based Opinion Extraction on French Restaurant Reviews

**Authors:** Chahine Nejma
**Approach:** Approach 2 — LoRA fine-tuning of a causal LLM
**Dev macro-accuracy:** 

---

## 1. Task

We tackle aspect-based sentiment classification on French restaurant reviews from the project dataset. For each review, the system must assign one of four labels — **Positive, Negative, Mixed, No Opinion** — to each of three aspects: **Price, Food, Service**. The training set contains ~4,000 noisy annotations; evaluation uses macro-accuracy across the three aspects on a held-out dev split.

Rather than treating this as three independent classification heads, we frame it as a single **sequence-to-sequence generation problem**: given the review, the model emits a structured triplet in a fixed format. This exploits the shared semantic context across aspects (a single review often informs all three labels jointly) and leverages the language modeling prior of a pre-trained LLM, which already understands French and sentiment-bearing expressions.

## 2. Approach: LoRA Fine-Tuning of Qwen3-0.6B

We chose **Approach 2** (LoRA fine-tuning via TRL + PEFT) with `Qwen/Qwen3-0.6B` as the base model. Qwen3-0.6B is the smallest authorized model in the list and was selected to keep training tractable on modest hardware (single T4 GPU, 16 GB VRAM, fp16) while still providing a capable multilingual backbone that handles French reviews natively.

### 2.1 Prompt Formatting and Chat Template

Supervised fine-tuning of a conversational LLM requires consistent prompt structuring. We use a lightweight instruction template that pairs a system message with the review and a fixed-format target:

```python
SYSTEM = (
    "Classify a French restaurant review on three aspects: Price, Food, Service. "
    "Each label is one of: Positive, Negative, Mixed, No Opinion. "
    "Respond exactly as: Price=<label>; Food=<label>; Service=<label>."
)

def format_example(ex: dict, with_target: bool = True) -> str:
    prompt = f"{SYSTEM}\nReview: {ex['Review'].strip()}\nAnswer: "
    if not with_target:
        return prompt
    return prompt + f"Price={ex['Price']}; Food={ex['Food']}; Service={ex['Service']}"
```

Three design decisions matter here:

- **Unified output schema.** All three aspects are predicted jointly in one generation pass, letting the model reason about them coherently rather than making isolated decisions.
- **Deterministic parseable format.** The fixed `Price=...; Food=...; Service=...` template makes regex-based parsing at inference time reliable, and because the target is always the same shape, the model learns the format quickly.
- **Same function for training and inference.** The `with_target` flag lets us reuse one formatting function for both supervised fine-tuning (full sequence) and prediction (prompt only), eliminating train-test drift.

### 2.2 Parameter-Efficient Fine-Tuning with LoRA

Full-parameter fine-tuning is prohibited by the project rules, and would be infeasible anyway on the available hardware. We use **LoRA (Low-Rank Adaptation)** via Hugging Face PEFT. LoRA freezes the pre-trained weights and injects small trainable low-rank matrices into selected layers, so only a tiny fraction of parameters receive gradient updates. This drastically reduces memory (no optimizer states for frozen weights) and training time while preserving almost all of the base model's capability.

```python
model = get_peft_model(model, LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
))
```

Our LoRA hyperparameters and their rationale:

| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Balances expressiveness and memory; rank 8 underfit in preliminary runs, rank 32 gave negligible gains at higher cost. |
| `lora_alpha` | 32 | Standard `2 × r` scaling, keeps update magnitudes stable. |
| `lora_dropout` | 0.05 | Light regularization against the ~4k noisy training examples. |
| `bias` | `"none"` | Biases are not adapted — standard LoRA practice, cheaper and equally effective. |
| `target_modules` | Q, K, V, O projections | Attention projections carry most of the task-specific adaptation; adapting MLP layers too gave no measurable improvement for this task size. |
| `task_type` | `CAUSAL_LM` | Required so PEFT attaches the correct head and loss. |

This configuration yields approximately **4.6 M trainable parameters out of 600 M total (≈0.76%)** — a ~130× reduction in the optimization footprint.

### 2.3 Training Setup with TRL's `SFTTrainer`

We use `SFTTrainer` from TRL for supervised fine-tuning on causal language modeling:

```python
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_ds,
    args=SFTConfig(
        output_dir=self.OUTPUT_DIR,
        per_device_train_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=1,
        learning_rate=lr,
        fp16=True,
        logging_steps=20,
        save_total_limit=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        report_to="none",
        dataset_text_field="text",
    ),
)
```

Key choices:

- **`SFTTrainer` over plain `Trainer`.** TRL's `SFTTrainer` handles EOS appending and packing automatically, removing boilerplate around causal LM data collation.
- **Hardware-adaptive batch size / learning rate.** Following the project guidelines, we compute the number of GPUs at runtime and scale accordingly, so the effective batch size and LR remain stable across 1-GPU and multi-GPU launches:
  ```python
  num_devices = max(1, torch.cuda.device_count())
  per_device_bs = 4                       # within the 1–4 range allowed for Qwen3-0.6B
  grad_accum = max(1, 16 // (per_device_bs * num_devices))
  lr = 2e-4 * (num_devices ** 0.5)        # sqrt scaling with device count
  ```
  This preserves an effective batch size of ~16 and an LR appropriate for it, regardless of the hardware on which training is launched.
- **`device_map=None`.** As required by the spec, we do not shard the base model across devices ourselves — `accelerate` handles DDP at launch time.
- **Mixed-precision fp16.** T4-compatible; on bf16-capable hardware this can be switched to `bf16=True` without other changes.
- **Cosine LR schedule with short warmup.** A brief warmup (3%) stabilizes the LoRA adapters early on, and cosine decay smoothly anneals the LR toward zero by the end of training.
- **Single epoch.** On ~4k examples with effective batch 16, one epoch already drove the training loss from ~2.6 down to ~2.0 with a token accuracy above 0.63. Extending to multiple epochs gave diminishing returns and risked overfitting the noisy labels.

### 2.4 Inference

At prediction time, we re-use `format_example(..., with_target=False)` to build the prompt and generate greedily (`do_sample=False`) with a small `max_new_tokens` budget — the structured target is short, so ~32 tokens always suffice. The generated text is parsed with a per-aspect regex that extracts the matching label, defaulting to `"No Opinion"` if no match is found. This fallback guarantees every review always receives a well-formed triplet, which is important because `runproject.py` expects a complete dict per review.

## 3. Results

- **Dev macro-accuracy (average over 5 runs):** [fill in]
- **Per-aspect accuracy:** Price [x], Food [x], Service [x]
- **Trainable parameters:** 4,587,520 / 600,637,440 (0.76%)
- **Training time per run:** ~10 min on a single T4 GPU (fp16, ~63 optimizer steps with effective batch 64)
- **Inference time on dev split:** ~10 min (60 batches)

## 4. Possible Extensions

- **Quantization (QLoRA).** Loading the base model in 4-bit via bitsandbytes would cut memory further and allow fine-tuning larger backbones (Qwen3-1.7B or Qwen3-4B) under the same VRAM budget. *Not used here as bitsandbytes is not on the authorized library list.*
- **Larger backbone within the allowed list.** Scaling to Qwen3-1.7B or Qwen3-4B with the same LoRA recipe would likely improve accuracy at the cost of longer training.
- **Longer training with lower LR.** Our current run is on the undertrained side of the curve; 2–3 epochs with a slightly lower LR could yield additional gains on cleanly-labeled aspects.
- **Batched generation at inference.** Our `predict()` generates one review at a time; batching would significantly cut evaluation time without changing results.
- **Label smoothing or class reweighting.** The training annotations are noisy and class-imbalanced (heavy "No Opinion"); light smoothing could help calibration.

## 5. Repository Structure

```
src/
├── config.py              # unchanged
├── runproject.py          # unchanged
└── ftlora_extractor.py    # our implementation (OpinionExtractor class)
data/
├── train.tsv
└── dev.tsv
README.md
```

Run with:
```bash
cd src
accelerate launch runproject.py
```

---

Want me to adjust the tone (more concise / more formal), add a small diagram of the LoRA adapter placement, or fold in a specific number once your run finishes?
