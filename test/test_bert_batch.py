import torch
from transformers import RobertaTokenizer, RobertaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

# Prepare two pairs of nl and code
pairs = [
    ("return maximum value", "def max(a,b): if a>b: return a else return b"),
    ("calculate sum", "def add(a,b): return a + b"),
    ("maximum value return", "if a>b: def max(a,b): else return b return a"),
    ("return maximum value", "def max(a,b): if a<b: return b else return a"),
]

# Construct batched input
batch_tokens = []
for nl, code in pairs:
    nl_tokens = tokenizer.tokenize(nl)
    code_tokens = tokenizer.tokenize(code)
    tokens = [tokenizer.cls_token] + nl_tokens + [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
    batch_tokens.append(tokens)

max_len = max(len(tks) for tks in batch_tokens)
batch_token_ids = [tokenizer.convert_tokens_to_ids(tks) + [tokenizer.pad_token_id] * (max_len-len(tks)) for tks in batch_tokens]
input_ids = torch.tensor(batch_token_ids, device=device)   # [batch_size, seq_len]
attention_mask = torch.tensor([[1]*len(tks)+[0]*(max_len-len(tks)) for tks in batch_tokens], device=device)

with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask)
    embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

# Output sentence embedding of each pair (e.g., at the [CLS] position, usually index 0)
for i, (nl, code) in enumerate(pairs):
    print(f"Pair {i+1}: \"{nl}\" + \"{code}\"")
    print("CLS embedding (first 5):", embeddings[i, 0, :5])
    print()