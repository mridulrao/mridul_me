import torch.nn.functional as F
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "./backend/models/LFM2.5-350M"

ADAPTER_PATH = (
    "./tunings/runs/"
    "lfm2_350m_hinglish_cpt_lora_vocab_extended_20260612_190348"
)

device = "cpu"


print("Loading extended tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Extended tokenizer vocab size: {len(tokenizer)}")


print("Loading original base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=False,
)

print("Original model embedding size:")
print(base_model.get_input_embeddings().weight.shape)


print("Resizing base model embeddings to match extended tokenizer...")
base_model.resize_token_embeddings(len(tokenizer))

# Useful if input/output embeddings are tied.
if hasattr(base_model, "tie_weights"):
    base_model.tie_weights()

print("Resized model embedding size:")
print(base_model.get_input_embeddings().weight.shape)


print("Loading PEFT adapter...")
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    device_map="cpu",
    is_trainable=False,
)

model.eval()

print("Model loaded successfully.")


def generate_hinglish(prompt, max_new=80):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            repetition_penalty=1.15,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    print(response)
    return response

def inspect_next_token(prompt, top_k=20):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits[0, -1]
        probs = F.softmax(logits, dim=-1)
        values, ids = torch.topk(probs, top_k)

    print(f"\nPrompt: {prompt}")
    print("Top next tokens:")

    for prob, tok_id in zip(values.tolist(), ids.tolist()):
        token = tokenizer.decode([tok_id])
        marker = ""

        if tok_id == tokenizer.eos_token_id:
            marker = "  <--- EOS"

        print(f"{tok_id:8d} | {prob:.4f} | {repr(token)}{marker}")

def inspect_prompt_tokens(prompt):
    ids = tokenizer.encode(prompt, add_special_tokens=False)

    print("\nPrompt:", repr(prompt))
    print("Full decoded:", repr(tokenizer.decode(ids)))
    print("Last tokens:")

    for tok_id in ids[-20:]:
        print(tok_id, repr(tokenizer.decode([tok_id])), tokenizer.convert_ids_to_tokens(tok_id))

inspect_prompt_tokens("Project deadline miss hone wali thi isliye manager ne")
inspect_prompt_tokens("Mummy ko bolna ki main late aaunga aur ")


# inspect_next_token("Mummy ko bolna ki main late aaunga aur")
# inspect_next_token("Project deadline miss hone wali thi isliye manager ne")