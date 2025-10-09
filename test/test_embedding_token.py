# Requires transformers>=4.51.0
from transformers import AutoTokenizer


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

# Each query must come with a one-sentence instruction that describes the task
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = [
    get_detailed_instruct(task, 'What is the capital of China?'),
    get_detailed_instruct(task, 'Explain gravity')
]
# No need to add instruction for retrieval documents
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun."
]
input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left')

max_length = 64

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    # padding=True,
    # padding="longest",
    padding="max_length",
    truncation=True,
    max_length=max_length,
    return_tensors="pt",
)

print("Special tokens map:", tokenizer.special_tokens_map)
print("Pad token id:", tokenizer.pad_token_id)
print("EOS token id:", getattr(tokenizer, "eos_token_id", None))


for i, txt in enumerate(input_texts):
    ids = batch_dict['input_ids'][i].tolist()
    toks = tokenizer.convert_ids_to_tokens(ids)
    print(f"\n--- Input #{i} ---\n{txt}")
    print(f"Tokens: {toks}")
    print(f"Token IDs: {ids}")


# 5) 单独查看 “Instruct:” 和 “Query:” 这两个 key word 的 token_ids
inst = tokenizer("Instruct:", add_special_tokens=False)
quer = tokenizer("Query:",    add_special_tokens=False)
print("\n>>> Guided Token IDs")
print("Instruct: ", inst['input_ids'], "→", tokenizer.convert_ids_to_tokens(inst['input_ids']))
print("Query:    ", quer['input_ids'], "→", tokenizer.convert_ids_to_tokens(quer['input_ids']))