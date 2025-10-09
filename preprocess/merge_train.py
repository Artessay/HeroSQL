import json
import torch
import random
import logging

from src.dataset import load_eval_dataset, load_plan_dataset
from src.utils import seed_everything



def run(mode: str):
    seed_everything(2025)

    # Load evaluation data 
    bird_data_list = load_eval_dataset("bird", mode)
    spider_data_list = load_eval_dataset("spider", mode)
    data_list = bird_data_list + spider_data_list

    bird_graph_list = load_plan_dataset("bird", mode)
    spider_graph_list = load_plan_dataset("spider", mode)
    graph_list = bird_graph_list + spider_graph_list

    # shuffle data
    index_list = list(range(len(data_list)))
    random.shuffle(index_list)

    data_list = [data_list[i] for i in index_list]
    graph_list = [graph_list[i] for i in index_list]

    # Construct dataset and save to disk
    save_path = f'data/processed/all-{mode}-eval.json'
    with open(save_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    save_path = f"data/processed/all-{mode}-plan.pt"
    torch.save(graph_list, save_path)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    run("train")
    run("dev")