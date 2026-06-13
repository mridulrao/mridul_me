import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


# ================== CONFIG ==================

BASE_MODEL = "./backend/models/LFM2.5-350M"

PREV_ADAPTER = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_cpt_lora_20260611_133452"
)

VOCAB_CPT_ADAPTER = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348"
)

DATA_PATH = "./data/mridul_sft_dataset.jsonl"

RUNS_DIR = Path("./tunings/runs")
RUN_NAME_PREFIX = "lfm2_350m_hinglish_sft_lora"

MAX_LENGTH = 512

# ============================================


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


def validate_messages(example):
    messages = example.get("messages")

    if not isinstance(messages, list):
        return False

    if len(messages) < 2:
        return False

    for msg in messages:
        if not isinstance(msg, dict):
            return False

        if msg.get("role") not in {"system", "user", "assistant"}:
            return False

        content = msg.get("content")
        if not isinstance(content, str):
            return False

        if not content.strip():
            return False

    return True


def weights_are_tied(model):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    if input_emb is None or output_emb is None:
        return False

    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()


@torch.no_grad()
def initialize_new_token_embeddings(
    model,
    base_tokenizer,
    extended_tokenizer,
    new_tokens,
):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    input_weight = input_emb.weight
    output_weight = output_emb.weight if output_emb is not None else None

    tied = weights_are_tied(model)
    device = input_weight.device

    initialized = 0

    for token in new_tokens:
        new_id = extended_tokenizer.convert_tokens_to_ids(token)

        old_piece_ids = base_tokenizer.encode(
            token,
            add_special_tokens=False,
        )

        if new_id is None or new_id < 0:
            continue

        if not old_piece_ids:
            continue

        old_piece_ids_tensor = torch.tensor(
            old_piece_ids,
            dtype=torch.long,
            device=device,
        )

        mean_embedding = input_weight[old_piece_ids_tensor].mean(dim=0)

        input_weight[new_id].copy_(mean_embedding)

        if output_weight is not None and not tied:
            output_weight[new_id].copy_(mean_embedding)

        initialized += 1

    print(f"Initialized {initialized} new token embeddings.")


def load_reconstructed_cpt_model_for_sft(
    base_model_path,
    prev_adapter_path,
    vocab_cpt_adapter_path,
    device,
):
    print("Loading extended tokenizer from vocab CPT adapter...")
    tokenizer = AutoTokenizer.from_pretrained(
        vocab_cpt_adapter_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Extended tokenizer vocab size: {len(tokenizer)}")

    print("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True,
    )

    if device == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    print("Loading original base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
    )

    print("Loading previous CPT adapter...")
    model = PeftModel.from_pretrained(
        model,
        prev_adapter_path,
        is_trainable=False,
    )

    print("Merging previous CPT adapter into base...")
    model = model.merge_and_unload()

    print("Resizing embeddings for extended tokenizer...")
    model.resize_token_embeddings(len(tokenizer))

    metadata_path = Path(vocab_cpt_adapter_path) / "added_tokens_metadata.json"

    if metadata_path.exists():
        print("Loading added token metadata...")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        new_tokens = metadata["actual_new_tokens"]

        initialize_new_token_embeddings(
            model=model,
            base_tokenizer=base_tokenizer,
            extended_tokenizer=tokenizer,
            new_tokens=new_tokens,
        )
    else:
        print("No added_tokens_metadata.json found. Skipping manual embedding initialization.")

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    print("Loading vocab-extended CPT adapter...")
    model = PeftModel.from_pretrained(
        model,
        vocab_cpt_adapter_path,
        is_trainable=False,
    )

    print("Merging vocab-extended CPT adapter into model...")
    model = model.merge_and_unload()

    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if device == "mps":
        print("Moving model to MPS...")
        model = model.to("mps")
    elif device == "cuda":
        print("Moving model to CUDA...")
        model = model.to("cuda")
    else:
        print("Using CPU.")

    print("Reconstructed CPT model ready for SFT.")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--prev-adapter", default=PREV_ADAPTER)
    parser.add_argument("--vocab-cpt-adapter", default=VOCAB_CPT_ADAPTER)
    parser.add_argument("--data", default=DATA_PATH)

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Use cpu for safest Mac training. Use mps if your setup supports it.",
    )

    args = parser.parse_args()

    run_name = f"{RUN_NAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    logging_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {output_dir}")

    print("Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files=args.data,
        split="train",
    )

    print("Filtering invalid message rows...")
    dataset = dataset.filter(validate_messages)

    dataset = dataset.train_test_split(
        test_size=0.05,
        seed=42,
    )

    print(
        f"Dataset ready: {len(dataset['train'])} train rows, "
        f"{len(dataset['test'])} eval rows"
    )

    model, tokenizer = load_reconstructed_cpt_model_for_sft(
        base_model_path=args.base_model,
        prev_adapter_path=args.prev_adapter,
        vocab_cpt_adapter_path=args.vocab_cpt_adapter,
        device=args.device,
    )

    print("Tokenizer eos_token:", repr(tokenizer.eos_token))
    print("Tokenizer pad_token:", repr(tokenizer.pad_token))
    print("Tokenizer chat_template exists:", tokenizer.chat_template is not None)

    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template. Either define tokenizer.chat_template "
            "or manually format messages into plain text."
        )

    sample_text = tokenizer.apply_chat_template(
        dataset["train"][0]["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    print("\nSample formatted training example:")
    print(sample_text[:1000])
    print("\n--- End sample ---\n")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "out_proj",
            "in_proj",
            "w1",
            "w3",
            "w2",
        ],
    )

    use_cpu = args.device == "cpu"

    training_args = SFTConfig(
        output_dir=str(output_dir),
        logging_dir=str(logging_dir),
        run_name=run_name,

        max_length=MAX_LENGTH,
        packing=False,

        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,

        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,

        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=args.device == "cuda",
        fp16=False,
        use_cpu=use_cpu,

        dataloader_pin_memory=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        report_to="tensorboard",
    )

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        peft_config=lora_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    trainer.add_callback(ProgressCallback())

    print("Starting SFT training...")
    trainer.train()

    print("Saving SFT LoRA adapter and tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"SFT LoRA adapter saved to: {output_dir}")
    print(f"TensorBoard: tensorboard --logdir {RUNS_DIR}")


if __name__ == "__main__":
    main()