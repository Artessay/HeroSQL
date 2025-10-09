from transformers import AutoModelForCausalLM, AutoTokenizer

# Model information
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    device_map="auto",            # Move model automatically to the correct device (GPU if available)
    torch_dtype="auto",           # Use the correct precision (fp16/bfloat16 if available)
    trust_remote_code=True,
)

# Example prompt
prompt = """### Instruction:
Write a Python function that computes the factorial of a number.

### Response:
"""

# Tokenize, generate, decode
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generate_ids = model.generate(
    **inputs,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    eos_token_id=tokenizer.eos_token_id
)
output = tokenizer.decode(generate_ids[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)

print(output)