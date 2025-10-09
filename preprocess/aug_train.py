import json
import torch
import random
import logging

from tqdm import tqdm
from src.analyzer import LogicalAnalyzer
from src.utils import seed_everything, augment_negative_sql

def shuffle_data_and_graph(data_list, graph_list):
    assert len(data_list) == len(graph_list)
    num_data = len(data_list)
    indices = list(range(num_data))
    random.shuffle(indices)
    shuffled_data_list = [data_list[i] for i in indices]
    shuffled_graph_list = [graph_list[i] for i in indices]
    return shuffled_data_list, shuffled_graph_list

def run(mode: str):
    seed_everything(2025)

    # Load dataset from disk
    data_path = f'data/processed/all-{mode}-eval.json'
    with open(data_path, 'r') as f:
        data_list = json.load(f)
    graph_path = f"data/processed/all-{mode}-plan.pt"
    graph_list = torch.load(graph_path, weights_only=False)

    assert len(data_list) == len(graph_list)
    num_data = len(data_list)
    print(f"Number of {mode} data: {num_data}")

    neg_data_list, neg_graph_list = [], []
    for data, graph in zip(data_list, graph_list):
        status = data['status']
        if status != "Success":
            neg_data_list.append(data)
            neg_graph_list.append(graph)

    num_aug_data = num_data - 2 * len(neg_data_list)
    neg_data_list, neg_graph_list = shuffle_data_and_graph(neg_data_list, neg_graph_list)
    print(f"Number of negative {mode} data: {len(neg_data_list)}")
    print(f"Number of augmented negative {mode} data needed: {num_aug_data}")

    # Create LogicalAnalyzer instance
    analyzer = LogicalAnalyzer()

    # Augment negative SQL for each data
    aug_data_list = []
    aug_graph_list = []
    for data in tqdm(neg_data_list, desc=f"Augmenting {mode} data"):
        if len(aug_data_list) >= num_aug_data:
            break

        status = data['status']
        assert status != "Success"

        sql = data['pred']
        negative_sql = augment_negative_sql(sql)
        
        if negative_sql is None:
            print("Failed to augment negative SQL, skip this data.")
            continue

        negative_plan = analyzer.get_query_plan(
            data['db_path'], 
            negative_sql
        )
        if negative_plan is None:
            print("Failed to get plan for negative SQL, skip this data.")
            continue
        
        negative_data = data.copy()
        negative_data['pred'] = negative_sql
        negative_data['status'] = "Semantic Error"  # Mark as negative sample

        aug_data_list.append(negative_data)
        aug_graph_list.append(negative_plan)

    assert len(aug_data_list) == len(aug_graph_list)
    num_aug_data = len(aug_data_list)
    print(f"Number of augmented {mode} data: {num_aug_data}")
    
    num_total_data = num_data + num_aug_data
    print(f"Total number of {mode} data after augmentation: {num_total_data}")

    total_data_indices = list(range(num_total_data))
    random.shuffle(total_data_indices)
    total_data_list = data_list + aug_data_list
    total_graph_list = graph_list + aug_graph_list
    total_data_list = [total_data_list[i] for i in total_data_indices]
    total_graph_list = [total_graph_list[i] for i in total_data_indices]

    # Save augmented dataset to disk
    save_path = f'data/processed/aug-{mode}-eval.json'
    with open(save_path, 'w') as f:
        json.dump(total_data_list, f, indent=4)
    save_path = f"data/processed/aug-{mode}-plan.pt"
    torch.save(total_graph_list, save_path)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    run("train")
    # run("dev")