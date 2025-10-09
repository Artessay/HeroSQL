import torch
from transformers import RobertaTokenizer, RobertaModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaModel.from_pretrained("microsoft/codebert-base")
model.to(device)

nl_tokens=tokenizer.tokenize("return maximum value")
code_tokens=tokenizer.tokenize("def max(a,b): if a>b: return a else return b")
print("Hello")

tokens=[tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]+code_tokens+[tokenizer.eos_token]
tokens_ids=tokenizer.convert_tokens_to_ids(tokens)

input_ids = torch.tensor(tokens_ids, device=device).unsqueeze(0)   # [1, seq_len]

with torch.no_grad():
    outputs = model(input_ids)
context_embeddings = outputs[0]
print(context_embeddings[0][0][:5])