import argparse
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer

# ================== CONFIG ==================
MODEL_NAME = "./tunings/runs/lfm2_350m_hinglish_cpt_lora_20260611_154930"
DATA_PATH = "./data/mridul_sft_dataset.jsonl"
RUNS_DIR = Path("./tunings/runs")
RUN_NAME_PREFIX = "lfm2_350m_hinglish_sft_lora"
# ===========================================

class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        parts = [f"step={state.global_step}"]
        if "epoch" in logs:
            parts.append(f"epoch={logs['epoch']:.2f}")
        if "loss" in logs:
            parts.append(f"train_loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        print(" | ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--cpt-adapter", default=None)
    parser.add_argument("--data", default=DATA_PATH)
    args = parser.parse_args()

    run_name = f"{RUN_NAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {output_dir}")

    # === Dataset ===
    dataset = load_dataset("json", data_files=args.data, split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # === Tokenizer ===
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === Model ===
    print("Loading model...")
    device_map = "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )

    if args.cpt_adapter:
        print(f"Merging CPT adapter: {args.cpt_adapter}")
        model = PeftModel.from_pretrained(model, args.cpt_adapter)
        model = model.merge_and_unload()

    # === LoRA (LFM2.5 specific) ===
    lora_config = LoraConfig(
        r=16,                    # Good balance for SFT
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w3", "w2"],
    )

    # === SFT Config ===
    training_args = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,

        max_length=384,      # Better for conversations
        packing=False,

        num_train_epochs=2,       # Start with 2, can increase
        per_device_train_batch_size=1,   # Safer on M4
        gradient_accumulation_steps=16,
        learning_rate=2e-5,       # Lower for SFT on top of CPT
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        bf16=torch.cuda.is_available(),
        fp16=False,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
    )

    # === Trainer with Chat Formatting ===
    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"], 
            tokenize=False, 
            add_generation_prompt=False
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,   # ← Critical for your data
    )

    trainer.add_callback(ProgressCallback())
    trainer.train()

    print("Saving model...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"SFT LoRA saved to: {output_dir}")


if __name__ == "__main__":
    main()