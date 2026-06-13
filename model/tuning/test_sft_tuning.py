import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


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

SFT_ADAPTER = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_sft_lora_20260613_014643"
)

DEVICE = "cpu"

# ============================================


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


def load_reconstructed_cpt_base():
    print("Loading SFT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        SFT_ADAPTER,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print("EOS token:", repr(tokenizer.eos_token))
    print("PAD token:", repr(tokenizer.pad_token))
    print("Chat template exists:", tokenizer.chat_template is not None)

    print("Loading base tokenizer...")
    base_tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )

    print("Loading original base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map=DEVICE,
        low_cpu_mem_usage=False,
    )

    print("Loading previous CPT adapter...")
    model = PeftModel.from_pretrained(
        model,
        PREV_ADAPTER,
        device_map=DEVICE,
        is_trainable=False,
    )

    print("Merging previous CPT adapter...")
    model = model.merge_and_unload()

    print("Resizing embeddings for extended tokenizer...")
    model.resize_token_embeddings(len(tokenizer))

    metadata_path = Path(VOCAB_CPT_ADAPTER) / "added_tokens_metadata.json"

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
        print("No added_tokens_metadata.json found. Skipping new token initialization.")

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    print("Loading vocab CPT adapter...")
    model = PeftModel.from_pretrained(
        model,
        VOCAB_CPT_ADAPTER,
        device_map=DEVICE,
        is_trainable=False,
    )

    print("Merging vocab CPT adapter...")
    model = model.merge_and_unload()

    return model, tokenizer


print("Reconstructing CPT base...")
base_model, tokenizer = load_reconstructed_cpt_base()

print("Loading SFT adapter...")
model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER,
    device_map=DEVICE,
    is_trainable=False,
)

model.eval()
print(f"Model loaded on {DEVICE}")


def build_prompt(prompt: str):
    system_prompt = (
        "You are Mridul Rao. Reply in casual Hinglish WhatsApp style. "
        "Keep replies short, natural, and conversational."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return text


def generate(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.6,
    top_p: float = 0.9,
    repetition_penalty: float = 1.08,
):
    text = build_prompt(prompt)

    print("\nFormatted prompt:")
    print(repr(text))

    inputs = tokenizer(
        text,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][input_len:]

    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    print("\n" + "=" * 60)
    print(f"User: {prompt}")
    print(f"Assistant: {response}")
    print("=" * 60)

    return response


if __name__ == "__main__":
    test_prompts = [
        "Have a good day!",
    ]

    for prompt in test_prompts:
        generate(prompt)