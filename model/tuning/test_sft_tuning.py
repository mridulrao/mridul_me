import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================== CONFIG ==================
MODEL_PATH = "./tunings/runs/lfm2_350m_hinglish_sft_lora_20260612_030242" 
DEVICE = "cpu"
# ===========================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
# Load base model + SFT adapter
base_model = AutoModelForCausalLM.from_pretrained(
    "./backend/models/LFM2.5-350M",   # Original base model
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=DEVICE,
    low_cpu_mem_usage=True,
)

model = PeftModel.from_pretrained(
    base_model,
    MODEL_PATH,
    device_map=DEVICE
)
model.eval()

print(f"Model loaded on {DEVICE}")

def generate(
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.75,
    top_p: float = 0.9,
    repetition_penalty: float = 1.12
):
    # Rich system prompt (customize as needed)
    system_prompt = """You are Mridul Rao. Reply in casual, natural Hinglish WhatsApp style.
Use emojis when it feels right. Keep replies short and conversational.
You can send multiple short messages separated by newlines."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's reply
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    print("\n" + "="*60)
    print(f"User: {prompt}")
    print(f"Assistant: {response}")
    print("="*60)
    return response


# ================== TEST CASES ==================
if __name__ == "__main__":
    test_prompts = [
        "Theek hai",
    ]

    for prompt in test_prompts:
        generate(prompt)