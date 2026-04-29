###### imports ##############""
from typing import Literal
import os
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
##########################################""

ASPECTS = ["Price", "Food", "Service"]

SYSTEM = (
    "Classify a French restaurant review on three aspects: Price, Food, Service. "
    "Each label is one of: Positive, Negative, Mixed, No Opinion. "
    "Respond exactly as: Price=<label>; Food=<label>; Service=<label>."
)

########## generation de prompts complet pour entrainer le models completement
def format_example(ex: dict, with_target: bool = True) -> str:
    prompt = f"{SYSTEM}\nReview: {ex['Review'].strip()}\nAnswer: "
    if not with_target:
        return prompt
    return prompt + f"Price={ex['Price']}; Food={ex['Food']}; Service={ex['Service']}"


class OpinionExtractor:
    # SET THE FOLLOWING CLASS VARIABLE to "FT" if you implemented a fine-tuning approach
    method: Literal["NOFT", "FT"] = "FT"
    BASE_MODEL_ID = "Qwen/Qwen3-1.7B" ### contrainte du PDF , aussi choix du model pour tester sur colab
    OUTPUT_DIR = "experiments"

    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def __init__(self, cfg) -> None:
        
        self.cfg = cfg
        
    # DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it   
    def train(self, train_data: list[dict], val_data: list[dict]) -> None:
        """
        Trains the model, if OpinionExtractor.method=="FT"
        """
        print("SAMPLE:", train_data[0])
        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.BASE_MODEL_ID, dtype=torch.bfloat16, device_map=None,
        )
        model.config.pad_token_id = tokenizer.pad_token_id

        model = get_peft_model(model, LoraConfig(
            r=32 , lora_alpha=64, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM",
            #target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            target_modules = "all-linear"
        ))
        model.print_trainable_parameters()

        train_ds = Dataset.from_list([{"text": format_example(ex)} for ex in train_data])

        num_devices = max(1, torch.cuda.device_count())
        grad_accum = max(1, 16 // num_devices)
        lr = 2e-4 * (num_devices ** 0.5)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=train_ds,
            args=SFTConfig(
    output_dir=self.OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=grad_accum,
    num_train_epochs=2,
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

        model.config.use_cache = False
        trainer.train()
        trainer.save_model(self.OUTPUT_DIR)
        tokenizer.save_pretrained(self.OUTPUT_DIR)

        self.model = trainer.model
        self.tokenizer = tokenizer
        self.model.config.use_cache = True
        self.model.eval()
        
# DO NOT MODIFY THE SIGNATURE OF THIS METHOD, add code to implement it
    def predict(self, texts: list[str]) -> list[dict]:
        """
        :param texts: list of reviews from which to extract the opinion values
        :return: a list of dicts, one per input review, containing the opinion values for the 3 aspects.
        """
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_ID)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            base = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_ID, dtype=torch.bfloat16, device_map=None,
            )
            self.model = PeftModel.from_pretrained(base, self.OUTPUT_DIR) \
                if os.path.isdir(self.OUTPUT_DIR) else base
            self.model.eval()

        device = next(self.model.parameters()).device
        results = []
        for review in texts:
            prompt = format_example({"Review": review}, with_target=False)
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    truncation=True, max_length=512).to(device)
            with torch.no_grad():
                gen = self.model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decoded = self.tokenizer.decode(
                gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            out = {a: "No Opinion" for a in ASPECTS}
            for a in ASPECTS:
                m = re.search(rf"{a}\s*=\s*(Positive|Negative|Mixed|No Opinion)", decoded)
                if m:
                    out[a] = m.group(1)
            results.append(out)
        return results
