import json
import re
from collections import Counter
from pathlib import Path

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


DATA_PATH = "./data/hinglish_chat_chunk.jsonl"
NEW_DATA_PATH = "./data/hinglish_cpt_combined.jsonl"

TOKENIZER_PATH = "./backend/models/LFM2.5-350M"
OUTPUT_PATH = "./data/hinglish_token_candidates.json"

MIN_TEXT_LENGTH = 50
MIN_FREQ = 50
MIN_TOKEN_SPLIT = 3
MAX_CANDIDATES = 2000

# Only keep normal romanized words
WORD_PATTERN = re.compile(r"[A-Za-z]+")


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


def load_and_prepare_dataset():
    print("Loading first dataset...")
    dataset = load_dataset(
        "json",
        data_files=DATA_PATH,
        split="train",
    )

    print("Loading second dataset...")
    new_dataset = load_dataset(
        "json",
        data_files=NEW_DATA_PATH,
        split="train",
    )

    print("Merging datasets...")
    dataset = concatenate_datasets([dataset, new_dataset])

    print("Formatting examples...")
    dataset = dataset.map(
        format_hinglish_example,
        remove_columns=dataset.column_names,
    )

    print("Filtering short/empty texts...")
    dataset = dataset.filter(
        lambda x: len(x["text"].strip()) > MIN_TEXT_LENGTH
    )

    return dataset


def count_words(dataset):
    counter = Counter()

    for row in dataset:
        text = row["text"].lower()
        words = WORD_PATTERN.findall(text)
        counter.update(words)

    return counter


def build_token_candidates(counter, tokenizer):
    vocab = tokenizer.get_vocab()

    candidates = []
    skipped_existing_vocab = 0
    skipped_low_freq = 0
    skipped_good_tokenization = 0

    for word, freq in counter.most_common():
        if freq < MIN_FREQ:
            skipped_low_freq += 1
            continue

        if word in vocab:
            skipped_existing_vocab += 1
            continue

        tokens = tokenizer.tokenize(word)
        num_tokens = len(tokens)

        if num_tokens < MIN_TOKEN_SPLIT:
            skipped_good_tokenization += 1
            continue

        candidates.append(
            {
                "word": word,
                "frequency": freq,
                "num_tokens": num_tokens,
                "tokens": tokens,
            }
        )

        if len(candidates) >= MAX_CANDIDATES:
            break

    stats = {
        "total_unique_words": len(counter),
        "num_candidates": len(candidates),
        "min_freq": MIN_FREQ,
        "min_token_split": MIN_TOKEN_SPLIT,
        "max_candidates": MAX_CANDIDATES,
        "skipped_existing_vocab": skipped_existing_vocab,
        "skipped_low_freq": skipped_low_freq,
        "skipped_good_tokenization": skipped_good_tokenization,
    }

    return candidates, stats


def main():
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_and_prepare_dataset()

    print(f"Dataset rows after filtering: {len(dataset)}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True,
    )

    print("Counting words...")
    counter = count_words(dataset)

    print("Building token candidates...")
    candidates, stats = build_token_candidates(counter, tokenizer)

    result = {
        "metadata": stats,
        "candidates": candidates,
    }

    print(f"Saving candidates to: {output_path}")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("\nTop 50 candidates:")
    for item in candidates[:50]:
        print(
            item["word"],
            item["frequency"],
            item["num_tokens"],
            item["tokens"],
        )

    print("\nStats:")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()