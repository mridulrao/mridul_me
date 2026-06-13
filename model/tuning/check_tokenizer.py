from transformers import AutoTokenizer

TOKENIZER_PATH = "./tunings/runs/lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348"

tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    trust_remote_code=True,
)

words = [
    "karke",
    "chahiye",
    "banao",
    "kyunki",
    "mujhe",
    "mushkil",
    "tumhe",
    "zaroori",
    "samajh",
    "hamesha",
]

for word in words:
    print("=" * 50)
    print("word:", word)
    print("tokens:", tokenizer.tokenize(word))
    print("ids:", tokenizer.encode(word, add_special_tokens=False))
    print("direct token id:", tokenizer.convert_tokens_to_ids(word))

# check wether adapter saved
from pathlib import Path
from safetensors.torch import load_file

ADAPTER_PATH = Path("./tunings/runs/lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348")
sd = load_file(ADAPTER_PATH / "adapter_model.safetensors")

print("Adapter keys:")
for k, v in sd.items():
    print(k, tuple(v.shape))


# compare token savings
import json

CANDIDATE_JSON = "./data/hinglish_token_candidates.json"

with open(CANDIDATE_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

candidates = data["candidates"]

for n in [500, 1000, 2000, 5000]:
    selected = candidates[:n]

    total_occurrences = sum(x["frequency"] for x in selected)
    saved_tokens = sum((x["num_tokens"] - 1) * x["frequency"] for x in selected)

    print("=" * 50)
    print("top_n:", n)
    print("total_occurrences:", total_occurrences)
    print("estimated_saved_tokens:", saved_tokens)