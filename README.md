# French Restaurant Reviews classification
## by Chahine NEJMA
**Authors:** Chahine Nejma
**Approach:** Approach 2 — LoRA fine-tuning
**Base model:** `Qwen3-0.6B and Qwen3-1.7B `

### Dev macro-accuracy

| Price | Food | Service | **Macro** |
|---|---|---|---|
| 88.01 | 86.5 | 88.33 | **87.60** |


<center>
<img src='https://onedrive.live.com/embed?resid=AE69638675180117%21292802&authkey=%21AO_qaECmI1InIyg&width=634&height=556' width="500">

*LoRA: Low-Rank Adaptation. Taken from the original LoRA paper.*
</center>

## 1. Task

We tackle aspect-based sentiment classification on French restaurant reviews from the project dataset. For each review, the system must assign one of four labels: **Positive, Negative, Mixed, No Opinion**, to each of three aspects: **Price, Food, Service**.

Rather than treating this as three independent classification heads, we frame it as a single **sequence-to-sequence generation problem**.

## 2. Approach: LoRA Fine-Tuning

We chose **Approach 2** (LoRA fine-tuning via TRL + PEFT) with `Qwen/Qwen3-0.6B` as the base model. Qwen3-0.6B is the smallest authorized model in the list and was selected to keep training tractable on modest hardware. We ran everything on **Google Colab's free tier (single T4 GPU, 16 GB VRAM, fp16)**. The small footprint also kept iteration cycles short (~10 min per training run).

Note on scope: the report was originally written for Qwen3-0.6B (macro 80.77). After hyperparameter tuning (r=32, all-linear targets) I scaled to Qwen3-1.7B with the same LoRA recipe and obtained macro 87.60, which is the result reported in the headline table. Sections 2 and 3.1 retain the 0.6B configuration and 5-run analysis as the original ablation; Section 3.1's comparison table shows both models side by side.

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
we have for this setting:

- **Unified output schema.** All three aspects are predicted jointly in one generation pass, letting the model reason about them coherently rather than making isolated decisions.
- **Deterministic parseable format.** The fixed `Price=...; Food=...; Service=...` template makes parsing at inference time reliable, and because the target is always the same shape, the model learns the format quickly.

### 2.2 Parameter-Efficient Fine-Tuning with LoRA

We use **Low-Rank Adaptation**, which freezes the pre-trained weights entirely and injects small trainable low-rank matrices into selected layers. This gives practical benefits: memory consumption drops sharply as the major bottleneck in our training is the inference time of the larger model(tiny comparatively to the backpropagation required for a full model gradient), and the resulting adapter is very small (~20 MB) and can be loaded on top of the unchanged base model for inference.

For our configuration, LoRA yields exactly **4,587,520 trainable parameters out of 600,637,440 total (0.7638%)**  a ~130× reduction in the optimization footprint.

**Our LoRA hyperparameters and their rationale:**
\\**first setting:**
| Parameter | Value | Rationale |
|---|---|---|
| `r` (rank) | 16 | Balances expressiveness and memory; a common sweet spot for instruction-style fine-tuning. |
| `lora_dropout` | 0.05 | Light regularization against the ~4k noisy training examples. |
| `target_modules` | Q, K, V, O projections | Only the attention projection matrices are adapted. |


**Why adapt only Q, K, V, and O?** In a transformer block, the attention projections define *what the model attends to*, while the MLP projections act more like a frozen feature bank of general linguistic knowledge. The original LoRA paper found that adapting attention projections captures most of the task-specific gain for downstream adaptation, and this has become standard practice (at least I have heard this repeatedly in multiple courses).

in later implementations and to get a boost in accuracy we chose the following final configuration on **Qwen1.7B** after some hyperparameter tuning.
\\**second setting:**
| Parameter | Value | 
|---|---|
| `r` (rank) | 32 |
| `lora_dropout` | 0.05 | 
| `target_modules` | all-linear |

## 3. Results

### 3.1 Evaluation over 5 runs for the Qwen0.6B+LoRA

| Run | Price | Food | Service | Macro Acc |
|-----|-------|------|---------|-----------|
| 1 | 79.83 | 80.50 | 82.17 | 80.83 |
| 2 | 79.67 | 81.33 | 81.33 | 80.78 |
| 3 | 79.33 | 81.67 | 81.50 | 80.83 |
| 4 | 79.33 | 81.33 | 80.83 | 80.50 |
| 5 | 79.50 | 81.33 | 81.83 | 80.89 |
| **Avg** | **79.53** | **81.23** | **81.53** | **80.77** |

The variance across runs is very tight (80.50–80.89), indicating a stable pipeline across random seeds.

To quantify the impact of LoRA fine-tuning, we evaluated the base Qwen3-0.6B model (without any fine-tuning) on the full training set using the same prompt template:

| Metric | Base model (QWen 0.6B) | QWen 0.6B + LoRA| Qwen 1.7B + Lora |
|---|---|---| ---|
| Price | 25.89% | 79.53% | 88% |
| Food | 77.48% | 81.23% |     86.5%|
| Service | 63.83% | 81.53% |88.3%|
| **Macro** | **55.73%** | **80.77%** |**87.6%**|

### 3.2 Summary

| Metric | Value |
|---|---|
| Trainable parameters for Qwen 0.6B+LoRA(1st setting) | 4,587,520 / 600,637,440 (0.76%) |
| Training time per run Qwen 0.6B+LoRA(1st setting) | ~10 min |
| Inference time per dev split | ~10 min (60 batches) |
| Trainable parameters for Qwen 1.7B+LoRA(2nd setting) | 34,865,152 / 1,755,440,128 (1.98%) |
| Training time per run for Qwen 1.7B+LoRA(2nd setting) | ~39 min |
| Inference time per dev split | ~12 min (60 batches) |

Training and inference take roughly the same time on the first setting because LoRA does not affect forward pass cost, each pass still runs through the full frozen base model. The adapter only adds a negligible overhead. This highlights the efficiency of the method: LoRA fine-tunes less than 1% of the parameters while achieving strong performance.

### 3.3 Base model vs. fine-tuned model: qualitative comparison

The following samples illustrate how the base Qwen3-0.6B (without fine-tuning) compares to the LoRA-fine-tuned model on the same inputs, and through the same promt behavior.
I was suprised by the fact that through simple prompting the baseline model was able to mimic the output format desired.

**Sample 1** — *"J'en ai marre car je ne trouve pas de critiques péjoratives! C'est une valeur sûr..."*
Raw out:  Price=No Opinion; Food=Positive; Service=Positive
Base model:  Price=Negative; Food=Positive; Service=Positive.

| | Price | Food | Service |
|---|---|---|---|
| True | No Opinion | Positive | Positive |
| Base model | Negative | Positive | Positive |
| Fine-tuned | No Opinion | Positive | Positive |


**Sample 2** — *"Très longue attente mais personnel agréable et sympathique. Velouté de légume e..."*
Raw out:  Price=No Opinion; Food=Negative; Service=Positive
Base model:  Price=Negative; Food=Positive; Service=Mixed.

| | Price | Food | Service |
|---|---|---|---|
| True | No Opinion | Negative | Mixed |
| Base model | Negative | Positive | Mixed |
| Fine-tuned | No Opinion | Negative | Positive |


**Sample 3** — *"Très jolie vue sur la baie. Service très lent, les 2 tables qui nous entouraient..."*
Raw out:  Price=No Opinion; Food=Positive; Service=Negative
Base model:  Price=Negative; Food=Positive; Service=Negative.

| | Price | Food | Service |
|---|---|---|---|
| True | No Opinion | Positive | Negative |
| Base model | Negative | Positive | Negative |
| Fine-tuned | No Opinion | Positive | Negative |


**Key observations:** The base model tends to default to "Negative" for Price even when the review expresses no opinion on pricing, suggesting it lacks the nuance to distinguish absence of opinion from negative sentiment. The fine-tuned model handles "No Opinion" reliably, which is consistent with its stronger Price accuracy (79.53% vs. the base model's tendency to over-predict Negative).

### 3.4 Output format compliance

The fine-tuned model outputs the exact expected format (`Price=<label>; Food=<label>; Service=<label>`) on 100% of evaluated samples. No parse failures were observed. Every prediction was genuinely produced by the model, not a fallback rule. This confirms that the SFT training successfully taught the model both the task semantics and the output structure.

## 4. Possible Extensions
- **Constrained decoding** Constrained decoding forces the model to only generate tokens that are compatible with a predefined output format. In our case, we verified format compliance on all evaluated samples and found zero deviations; the SFT training already taught the model the exact output structure. Therefore, constrained decoding would add complexity with no accuracy benefit.

- **Quantization (QLoRA).** Loading the base model in 4-bit via bitsandbytes would cut memory further and allow fine-tuning larger backbones (Qwen3-1.7B or Qwen3-4B) under the same VRAM budget. *Not used here as bitsandbytes is not on the authorized library list.*

    <center>
    <img src='https://onedrive.live.com/embed?resid=AE69638675180117%21292801&authkey=%21AIBM2HNKRF7tzGo&width=1980&height=866' width="700">

    *QLoRA. Taken from the original QLoRA paper.*
    </center>

- **Larger backbone within the allowed list.** Scaling to Qwen3-4B with the same LoRA recipe would likely improve accuracy at the cost of longer training.

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
