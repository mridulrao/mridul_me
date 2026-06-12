from collections import Counter
from transformers import AutoModelForCausalLM

MODEL_NAME = "./backend/models/LFM2.5-350M"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="cpu",
)

print("\n=== Model class ===")
print(type(model))

print("\n=== Full model architecture ===")
print(model)

print("\n=== Linear module names ===")
linear_modules = []

for name, module in model.named_modules():
    class_name = module.__class__.__name__

    if class_name == "Linear":
        linear_modules.append(name)
        print(name)

print("\n=== Unique final module names for LoRA target_modules ===")
final_names = [name.split(".")[-1] for name in linear_modules]
for name, count in Counter(final_names).most_common():
    print(f"{name}: {count}")

print("\n=== Attention-like modules ===")
for name, module in model.named_modules():
    lname = name.lower()
    cname = module.__class__.__name__.lower()

    if any(k in lname or k in cname for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj"]):
        print(f"{name} -> {module.__class__.__name__}")

print("\n=== Conv-like modules ===")
for name, module in model.named_modules():
    cname = module.__class__.__name__.lower()
    lname = name.lower()

    if "conv" in cname or "conv" in lname:
        print(f"{name} -> {module.__class__.__name__}")

print("\n=== MLP / FFN-like modules ===")
for name, module in model.named_modules():
    lname = name.lower()

    if any(k in lname for k in ["mlp", "ffn", "feed_forward", "gate", "up_proj", "down_proj"]):
        print(f"{name} -> {module.__class__.__name__}")