import torch
from src.analyzer import RelGraphUtils

save_path = "data/processed/all-dev-plan.pt"
dataset = torch.load(save_path, weights_only=False)
print(f"Loaded dataset size: {len(dataset)}")
print(f"Loaded first sample: {dataset[0]}")
RelGraphUtils.print_graph(dataset[0])