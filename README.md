# French Restaurant Reviews classification

\\ **Authors:** Chahine Nejma
\\ **Approach:** Approach 2 — LoRA fine-tuning
\\ **Base model:** `Qwen/Qwen3-0.6B`

### Dev macro-accuracy

| Price | Food | Service | **Macro** |
|---|---|---|---|
| 79.67 | 81.33 | 82.00 | **81.00** |

<center>
<img src='https://onedrive.live.com/embed?resid=AE69638675180117%21292802&authkey=%21AO_qaECmI1InIyg&width=634&height=556' width="500">

*LoRA: Low-Rank Adaptation. Taken from the original LoRA paper.*
</center>

## 1. Task

We tackle aspect-based sentiment classification on French restaurant reviews from the project dataset. For each review, the system must assign one of four labels: **Positive, Negative, Mixed, No Opinion**, to each of three aspects: **Price, Food, Service**.

Rather than treating this as three independent classification heads, we frame it as a single **sequence-to-sequence generation problem**: given the review, the model emits a structured triplet in a fixed format.

## 2. Approach: LoRA Fine-Tuning of Qwen3-0.6B

We chose **Approach 2** (LoRA fine-tuning via TRL + PEFT) with `Qwen/Qwen3-0.6B` as the base model. Qwen3-0.6B is the smallest authorized model in the list and was selected to keep training tractable on modest hardware we ran everything on **Google Colab's free tier (single T4 GPU, 16 GB VRAM, fp16)** while still providing a capable multilingual backbone that handles French reviews natively. The small footprint also kept iteration cycles short (~10 min per training run).

### 2.1 Prompt Formatting and Chat Template

Supervised fine-tuning of a causal LLM requires consistent prompt structuring so that the model learns a stable input→output mapping. We use a lightweight instruction template that pairs a system message with the review and a fixed-format target triplet:

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
- **Deterministic parseable format.** The fixed `Price=...; Food=...; Service=...` template makes parsing at inference time reliable, and because the target is always the same shape, the model learns the format quickly.
- **Same function for training and inference.** The `with_target` flag lets us reuse one formatting function for both supervised fine-tuning (full sequence) and prediction (prompt only).

### 2.2 Parameter-Efficient Fine-Tuning with LoRA

We use **Low-Rank Adaptation**, which freezes the pre-trained weights entirely and injects small trainable low-rank matrices into selected layers. This gives three practical benefits: memory consumption drops sharply as the major bottle neck in our training is the inference time of the larger model(tiny comparatively to the backpropagation required for a full model gradient), and the resulting adapter is very small (~20 MB) and can be loaded on top of the unchanged base model for inference.

For our configuration, LoRA yields exactly **4,587,520 trainable parameters out of 600,637,440 total (0.7638%)**  a ~130× reduction in the optimization footprint.

**Our LoRA hyperparameters and their rationale:**

| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Balances expressiveness and memory; a common sweet spot for instruction-style fine-tuning. |
| `lora_dropout` | 0.05 | Light regularization against the ~4k noisy training examples. |
| `target_modules` | Q, K, V, O projections | Only the attention projection matrices are adapted. |

**Why adapt only Q, K, V, and O?** In a transformer block, the attention projections define *what the model attends to*, while the MLP projections act more like a frozen feature bank of general linguistic knowledge. The original LoRA paper found that adapting attention projections captures most of the task-specific gain for downstream adaptation, and this has become standard practice (at least I have heard this repeatedly in multiple classes).


## 3. Results

| Metric | Value |
|---|---|
| **Dev macro-accuracy** | **81.00%** |
| Price accuracy | 79.67% |
| Food accuracy | 81.33% |
| Service accuracy | 82.00% |
| Trainable parameters | 4,587,520 / 600,637,440 (0.76%) |
| Training time per run | ~10 min |
| Inference time per dev split | ~10 min (60 batches) |

I find it interesting that training and inference took roughly the same time. This isn't really a LoRA effect. LoRA has no effect on the forward pass cost, since each pass still runs through the full frozen base model.
## 4. Possible Extensions

- **Quantization (QLoRA).** Loading the base model in 4-bit via bitsandbytes would cut memory further and allow fine-tuning larger backbones (Qwen3-1.7B or Qwen3-4B) under the same VRAM budget. *Not used here as bitsandbytes is not on the authorized library list.*

    <center>
    <img src='https://onedrive.live.com/embed?resid=AE69638675180117%21292801&authkey=%21AIBM2HNKRF7tzGo&width=1980&height=866' width="700">

    *QLoRA. Taken from the original QLoRA paper.*
    </center>

- **Larger backbone within the allowed list.** Scaling to Qwen3-1.7B or Qwen3-4B with the same LoRA recipe would likely improve accuracy at the cost of longer training.

- **Longer training with lower LR.** Our current run sits on the undertrained side of the loss curve; 2–3 epochs with a slightly lower LR could yield additional gains, particularly on the cleaner aspects.

- **Encoder + linear classifier baseline.** Since the task is fundamentally a classification problem, an alternative direction would be to drop generation altogether and instead extract sentence embeddings from a frozen encoder (e.g. a French BERT variant from Approach 3) and train lightweight linear heads on top. This would disregard the sequence-generation machinery entirely and focus purely on the classification aspect.

## 5. Repository Structure

```
src/
├── config.py              
├── runproject.py          
└── ftlora_extractor.py    # our implementation 
data/
├── train.tsv
└── dev.tsv
README.md
```
