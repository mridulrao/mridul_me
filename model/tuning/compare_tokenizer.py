import re
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

BASE_TOKENIZER = "./backend/models/LFM2.5-350M"
NEW_TOKENIZER = "./tunings/runs/lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348"

DATA_PATH = "./data/hinglish_chat_chunk.jsonl"
NEW_DATA_PATH = "./data/hinglish_cpt_combined.jsonl"


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

    return {"text": "\n\n".join(parts)}


def load_data():
    d1 = load_dataset("json", data_files=DATA_PATH, split="train")
    d2 = load_dataset("json", data_files=NEW_DATA_PATH, split="train")
    dataset = concatenate_datasets([d1, d2])

    dataset = dataset.map(
        format_hinglish_example,
        remove_columns=dataset.column_names,
    )

    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 50)
    return dataset


def tokenizer_stats(dataset, tokenizer, max_rows=5000):
    total_words = 0
    total_tokens = 0
    total_chars = 0

    for row in dataset.select(range(min(len(dataset), max_rows))):
        text = row["text"].lower()
        words = re.findall(r"[a-z]+", text)

        input_ids = tokenizer.encode(text, add_special_tokens=False)

        total_words += len(words)
        total_tokens += len(input_ids)
        total_chars += len(text)

    return {
        "tokens_per_word": total_tokens / max(total_words, 1),
        "tokens_per_char": total_tokens / max(total_chars, 1),
        "total_words": total_words,
        "total_tokens": total_tokens,
        "total_chars": total_chars,
    }


dataset = load_data()

old_tok = AutoTokenizer.from_pretrained(BASE_TOKENIZER, trust_remote_code=True)
new_tok = AutoTokenizer.from_pretrained(NEW_TOKENIZER, trust_remote_code=True)

print("Old tokenizer:")
print(tokenizer_stats(dataset, old_tok))

print("\nNew tokenizer:")
print(tokenizer_stats(dataset, new_tok))