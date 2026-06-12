from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model


MODEL_NAME = "tunings/runs/lfm2_350m_hinglish_cpt_lora_20260611_133452" # "./backend/models/LFM2.5-350M" (previous run)

DATA_PATH = "./data/hinglish_chat_chunk.jsonl"
NEW_DATA_PATH = "./data/hinglish_cpt_combined.jsonl"

RUNS_DIR = Path("./tunings/runs")
RUN_NAME_PREFIX = "lfm2_350m_hinglish_cpt_lora"

MAX_LENGTH = 512

def format_hinglish_example(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output = (example.get("output") or "").strip()

    parts = []

    if instruction:
        parts.append(instruction)

    if input_text:
        parts.append(input_text)

    if output:
        parts.append(output)

    return {
        "text": "\n\n".join(parts)
    }

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


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated["input_ids"])

    total_length = (total_length // MAX_LENGTH) * MAX_LENGTH

    result = {
        k: [
            t[i : i + MAX_LENGTH]
            for i in range(0, total_length, MAX_LENGTH)
        ]
        for k, t in concatenated.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


def main():
    run_name = f"{RUN_NAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    logging_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {output_dir}")
    print(f"TensorBoard logs: {logging_dir}")

    print("Loading raw text dataset...")
    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train",
    )

    print("Loading new raw text dataset...")
    new_dataset = load_dataset(
        "json",
        data_files=NEW_DATA_PATH,
        split="train",
    )

    print("Merging the datasets...")
    dataset = concatenate_datasets([dataset, new_dataset])

    dataset = dataset.map(
        format_hinglish_example,
        remove_columns=dataset.column_names,
    )

    # Filter out empty or very short texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 50)

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    print(
        f"Dataset ready: {len(dataset['train'])} train text rows, "
        f"{len(dataset['test'])} eval text rows"
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = dataset.map(
        lambda example: {
            "text": example["text"].strip() + tokenizer.eos_token
        },
        desc="Adding EOS tokens",
    )

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    print("Grouping tokens into fixed-length blocks...")
    lm_dataset = tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping into {MAX_LENGTH}-token blocks",
    )

    print(
        f"LM dataset ready: {len(lm_dataset['train'])} train blocks, "
        f"{len(lm_dataset['test'])} eval blocks"
    )

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
        r=24, #16 (previous run), # (rank 16-664 common for CPT)
        lora_alpha=48, #32 (previous run), # (typically 2*r)
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",      # Covers both attention and conv blocks
            "in_proj",       # Conv blocks
            "w1",            # MLP (gate/up equivalent)
            "w3",            # MLP (up/gate equivalent)
            "w2",            # MLP (down)
        ],
        # modules_to_save=["lm_head"]   # Optional, only if you see vocabulary issues (increase memory issue and unstable training)
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    print("Building training configuration...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        run_name=run_name,

        # Training
        num_train_epochs=2,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=3e-5, # 3e-5 (checkpoint1) #2e-4(previous run), # or 1e-4 ~ 3e-4 range
        lr_scheduler_type="cosine",  # smooth decay
        warmup_ratio=0.05, # earlier 0.03
        weight_decay=0.01,
        #resume_from_checkpoint=str("./tunings/runs/lfm2_350m_hinglish_cpt_lora_20260611_133452"),

        # Logging / saving
        logging_steps=1,
        logging_strategy="steps",
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Precision
        bf16=torch.cuda.is_available(),
        fp16=False,

        # Optimizer
        optim="adamw_torch",

        # Misc
        report_to="tensorboard",
        remove_unused_columns=False,
        disable_tqdm=False,
    )

    print("Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.add_callback(ProgressCallback())

    print("Starting CPT training...")
    trainer.train()

    print("Saving CPT LoRA adapter and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved CPT LoRA adapter to {output_dir}")
    print(f"Open TensorBoard with: tensorboard --logdir {RUNS_DIR}")


if __name__ == "__main__":
    main()
