import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "./backend/models/LFM2.5-350M"

PREV_ADAPTER = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_cpt_lora_20260611_133452"
)

CURRENT_ADAPTER = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348"
)

ADDED_TOKENS_METADATA = Path(CURRENT_ADAPTER) / "added_tokens_metadata.json"

device = "cpu"


def weights_are_tied(model):
    input_emb = model.get_input_embeddings()
    output_emb = model.get_output_embeddings()

    if input_emb is None or output_emb is None:
        return False

    return input_emb.weight.data_ptr() == output_emb.weight.data_ptr()


@torch.no_grad()
def initialize_new_token_embeddings(model, base_tokenizer, extended_tokenizer, new_tokens):
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


print("Loading extended tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    CURRENT_ADAPTER,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Extended vocab size: {len(tokenizer)}")


print("Loading base tokenizer...")
base_tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)


print("Loading original base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=False,
)


print("Loading previous adapter and merging it...")
base_model = PeftModel.from_pretrained(
    base_model,
    PREV_ADAPTER,
    device_map="cpu",
    is_trainable=False,
)

base_model = base_model.merge_and_unload()
print("Previous adapter merged.")


print("Resizing embeddings to extended tokenizer...")
base_model.resize_token_embeddings(len(tokenizer))


print("Loading added token metadata...")
with open(ADDED_TOKENS_METADATA, "r", encoding="utf-8") as f:
    metadata = json.load(f)

new_tokens = metadata["actual_new_tokens"]

print(f"New tokens from metadata: {len(new_tokens)}")


print("Initializing new token embeddings exactly like training...")
initialize_new_token_embeddings(
    model=base_model,
    base_tokenizer=base_tokenizer,
    extended_tokenizer=tokenizer,
    new_tokens=new_tokens,
)

if hasattr(base_model, "tie_weights"):
    base_model.tie_weights()


print("Loading current vocab-extended adapter...")
model = PeftModel.from_pretrained(
    base_model,
    CURRENT_ADAPTER,
    device_map="cpu",
    is_trainable=False,
)

model.eval()
print("Model loaded successfully.")


def generate_hinglish(prompt, max_new=80):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            min_new_tokens=20,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            repetition_penalty=1.08,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    print("\nPROMPT:")
    print(prompt)
    print("\nGENERATED:")
    print(generated_text)
    return generated_text


generate_hinglish("Project deadline miss hone wali thi isliye manager ne team ko bola ki ")
generate_hinglish("Mummy ko bolna tha ki main late aaunga kyunki traffic ")
generate_hinglish("Customer gussa tha kyunki uska issue solve nahi hua tha, toh maine ")