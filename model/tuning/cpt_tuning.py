from datetime import datetime
from pathlib import Path
import json

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from tokenizers import AddedToken
from peft import LoraConfig, get_peft_model, PeftModel


BASE_MODEL = "./backend/models/LFM2.5-350M"
PREV_ADAPTER = "./tunings/runs/lfm2_350m_hinglish_cpt_lora_20260611_133452"

DATA_PATH = "./data/hinglish_chat_chunk.jsonl"
NEW_DATA_PATH = "./data/hinglish_cpt_combined.jsonl"
TOKEN_CANDIDATES_PATH = "./data/hinglish_token_candidates.json"

RUNS_DIR = Path("./tunings/runs")
RUN_NAME_PREFIX = "lfm2_350m_hinglish_cpt_lora_vocab_extended"

MAX_LENGTH = 512

# Start small. Try 500 first, then 1000/2000 later.
MAX_NEW_TOKENS = 500

USE_PREV_ADAPTER = True


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


def load_token_candidates(path, max_new_tokens):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Supports:
    # {"candidates": [{"word": "karke", ...}, ...]}
    # or ["karke", "chahiye", ...]
    if isinstance(data, dict):
        candidates = data.get("candidates", [])
    elif isinstance(data, list):
        candidates = data
    else:
        raise ValueError("Unsupported token candidate JSON format.")

    words = []

    for item in candidates:
        if isinstance(item, dict):
            word = item.get("word")
        else:
            word = item

        if not word:
            continue

        word = str(word).strip().lower()

        if word and word not in words:
            words.append(word)

        if len(words) >= max_new_tokens:
            break

    return words


def find_module_name(model, target_module):
    for name, module in model.named_modules():
        if module is target_module:
            return name
    return None


def weights_are_tied(model):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    if input_emb is None or output_emb is None:
        return False

    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()


@torch.no_grad()
def initialize_new_token_embeddings(model, tokenizer, new_tokens, old_piece_ids_by_token):
    """
    Instead of random initialization, initialize each new token embedding as
    the average of the embeddings of the old tokens it used to split into.

    Example:
      "karke" used to be ["k", "ark", "e"]
      new embedding("karke") = mean(embedding("k"), embedding("ark"), embedding("e"))
    """
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    if input_emb is None:
        raise ValueError("Model does not expose input embeddings.")

    input_weight = input_emb.weight
    device = input_weight.device

    output_weight = output_emb.weight if output_emb is not None else None
    tied = weights_are_tied(model)

    initialized = 0

    for token in new_tokens:
        new_id = tokenizer.convert_tokens_to_ids(token)
        old_piece_ids = old_piece_ids_by_token.get(token, [])

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

        # If lm_head is not tied to input embeddings, initialize its row too.
        if output_weight is not None and not tied:
            output_weight[new_id].copy_(mean_embedding)

        initialized += 1

    print(f"Initialized {initialized} new token embeddings using old split-token averages.")


def build_trainable_token_indices(model, new_token_ids):
    """
    PEFT supports either:
      trainable_token_indices=[ids]
    or:
      trainable_token_indices={"embed_tokens": [ids], "lm_head": [ids]}

    For safety, this function finds the actual embedding module names.
    """
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    input_name = find_module_name(model, input_emb)
    output_name = find_module_name(model, output_emb) if output_emb is not None else None

    print(f"Input embedding module: {input_name}")
    print(f"Output embedding module: {output_name}")
    print(f"Weights tied: {weights_are_tied(model)}")

    if input_name is None:
        print("Could not find input embedding module name. Falling back to list format.")
        return new_token_ids

    trainable = {
        input_name: new_token_ids,
    }

    if output_name is not None and output_name != input_name:
        trainable[output_name] = new_token_ids

    return trainable


def main():
    run_name = f"{RUN_NAME_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = RUNS_DIR / run_name
    logging_dir = output_dir / "logs"

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {output_dir}")
    print(f"TensorBoard logs: {logging_dir}")

    print("Loading tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading token candidates...")
    candidate_tokens = load_token_candidates(
        TOKEN_CANDIDATES_PATH,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print(f"Candidate tokens loaded: {len(candidate_tokens)}")

    # Save old split-token ids before adding new tokens.
    old_piece_ids_by_token = {}
    for token in candidate_tokens:
        old_piece_ids_by_token[token] = tokenizer.encode(
            token,
            add_special_tokens=False,
        )

    old_vocab_size = len(tokenizer)
    print(f"Old tokenizer vocab size: {old_vocab_size}")

    print("Adding new Hinglish tokens to tokenizer...")
    added_tokens = [
        AddedToken(
            token,
            single_word=True,
            normalized=False,
        )
        for token in candidate_tokens
    ]

    num_added = tokenizer.add_tokens(added_tokens)
    print(f"Actually added new tokens: {num_added}")

    new_vocab_size = len(tokenizer)
    print(f"New tokenizer vocab size: {new_vocab_size}")

    actual_new_tokens = []
    new_token_ids = []

    for token in candidate_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)

        if token_id is None:
            continue

        # Only keep tokens that were truly added to the end of vocab.
        if token_id >= old_vocab_size:
            actual_new_tokens.append(token)
            new_token_ids.append(token_id)

    print(f"Actual new trainable tokens: {len(actual_new_tokens)}")

    if not actual_new_tokens:
        raise ValueError("No new tokens were added. Check your token candidate JSON.")

    print("Saving added token metadata...")
    with open(output_dir / "added_tokens_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "old_vocab_size": old_vocab_size,
                "new_vocab_size": new_vocab_size,
                "num_added": num_added,
                "actual_new_tokens": actual_new_tokens,
                "new_token_ids": new_token_ids,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

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

    print("Merging datasets...")
    dataset = concatenate_datasets([dataset, new_dataset])

    dataset = dataset.map(
        format_hinglish_example,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 50)

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    print(
        f"Dataset ready: {len(dataset['train'])} train text rows, "
        f"{len(dataset['test'])} eval text rows"
    )

    dataset = dataset.map(
        lambda example: {
            "text": example["text"].strip() + tokenizer.eos_token
        },
        desc="Adding EOS tokens",
    )

    print("Tokenizing dataset with extended tokenizer...")
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

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if USE_PREV_ADAPTER:
        print("Loading previous LoRA adapter...")
        model = PeftModel.from_pretrained(
            model,
            PREV_ADAPTER,
            is_trainable=False,
        )

        print("Merging previous LoRA adapter into base model...")
        model = model.merge_and_unload()

    print("Resizing model embeddings for extended tokenizer...")
    model.resize_token_embeddings(len(tokenizer))

    print("Initializing new token embeddings...")
    initialize_new_token_embeddings(
        model=model,
        tokenizer=tokenizer,
        new_tokens=actual_new_tokens,
        old_piece_ids_by_token=old_piece_ids_by_token,
    )

    # Useful for gradient checkpointing with PEFT.
    model.config.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print("Building trainable token indices...")
    trainable_token_indices = build_trainable_token_indices(
        model=model,
        new_token_ids=new_token_ids,
    )

    print("Configuring fresh LoRA + trainable token embeddings...")
    lora_kwargs = dict(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
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
        trainable_token_indices=trainable_token_indices,
    )

    try:
        lora_config = LoraConfig(
            **lora_kwargs,
            ensure_weight_tying=True,
        )
    except TypeError:
        print(
            "Your PEFT version may not support ensure_weight_tying. "
            "Continuing without it."
        )
        lora_config = LoraConfig(**lora_kwargs)

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

        # For LoRA + new token embeddings, 3e-5 is usually too slow.
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,

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

    print("Saving CPT LoRA adapter and extended tokenizer...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"Saved CPT LoRA adapter and tokenizer to {output_dir}")
    print(f"Open TensorBoard with: tensorboard --logdir {RUNS_DIR}")


if __name__ == "__main__":
    main()