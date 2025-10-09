from torch_geometric.data import Data, InMemoryDataset

class QueryPlanDataset(InMemoryDataset):
    def __init__(self, data_list: list[Data]):
        super().__init__()
        self.data, self.slices = self.collate(data_list)

if __name__ == "__main__":
    import os
    import torch

    # Create example Data objects
    data1 = Data(x=torch.tensor([[1.0], [2.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    data2 = Data(x=torch.tensor([[3.0], [4.0], [5.0]]), edge_index=torch.tensor([[0, 1], [1, 2]]))

    # Create the dataset
    dataset = QueryPlanDataset(data_list=[data1, data2])

    # Dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"First sample: {dataset[0]}")
    print(f"Second sample: {dataset[1]}")
    
    # Save dataset
    save_path = "custom_geometric_dataset.pt"
    torch.save(dataset, save_path)

    # Load dataset
    loaded_dataset = torch.load(save_path, weights_only=False)
    print(f"Loaded dataset size: {len(loaded_dataset)}")
    print(f"Loaded first sample: {loaded_dataset[0]}")
    print(f"Loaded second sample: {loaded_dataset[1]}")

    # Delete dataset
    os.remove(save_path)