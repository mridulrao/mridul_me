import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME = "./backend/models/LFM2.5-350M"
DATA_PATH = "/backend/data/mridul_sft_dataset.jsonl"
OUTPUT_DIR = "./backend/tunings/lfm2_350m_mridul_lora"


class ProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return

        parts = [f"step={state.global_step}"]

        epoch = logs.get("epoch")
        if epoch is not None:
            parts.append(f"epoch={epoch:.2f}")

        loss = logs.get("loss")
        if loss is not None:
            parts.append(f"train_loss={loss:.4f}")

        eval_loss = logs.get("eval_loss")
        if eval_loss is not None:
            parts.append(f"eval_loss={eval_loss:.4f}")

        learning_rate = logs.get("learning_rate")
        if learning_rate is not None:
            parts.append(f"lr={learning_rate:.2e}")

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            parts.append(f"grad_norm={grad_norm:.4f}")

        print(" | ".join(parts))

def main():
    print("Loading dataset...")
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    print(
        f"Dataset ready: {len(dataset['train'])} train samples, "
        f"{len(dataset['test'])} eval samples"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    print("Model loaded.")

    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print("Building training configuration...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,

        # Dataset
        max_length=512,
        packing=False,

        # Training
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,

        # Logging / saving
        logging_steps=1,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision
        bf16=torch.cuda.is_available(),
        fp16=False,

        # Optimizer
        optim="adamw_torch",

        # Misc
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.add_callback(ProgressCallback())

    print("Starting training...")
    trainer.train()

    print("Saving adapter and tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved LoRA adapter to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
