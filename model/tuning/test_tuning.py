from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"
torch.set_default_device(device)   # Important for new tensors

MODEL_PATH = "./tunings/runs/lfm2_350m_hinglish_cpt_lora_20260611_154930"
# MODEL_PATH = "./backend/models/LFM2.5-350M"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading model...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,                    # Since it's already a PEFT checkpoint
    trust_remote_code=True,
    torch_dtype=torch.float32,     # Use float32 on CPU (safer, though slower)
    device_map="cpu",              # Force CPU
    low_cpu_mem_usage=True,
)

# If the above doesn't work well, use this instead:
# model = PeftModel.from_pretrained(base_model, MODEL_PATH, device_map="cpu")

model = base_model  # Already includes adapter if you loaded the PEFT dir
model.eval()

def generate_hinglish(prompt, max_new=30, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Clean up the prompt repetition if needed
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    print(response)
    return response


# Test cases
print("=== Test 1 ===")
generate_hinglish("Aur batao sab bdiya hai?")

# print("\n=== Test 2 ===")
# generate_hinglish("Yeh code thoda optimize kar do yaar:")

# print("\n=== Test 3 ===")
# generate_hinglish("Mummy ko bolna ki main late aaunga, traffic bahut hai.")