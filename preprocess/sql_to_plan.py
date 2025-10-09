import json
import torch
import logging
from tqdm import tqdm

from src.analyzer import LogicalAnalyzer
from src.utils import seed_everything



def run(args):
    seed_everything(2025)

    # Load evaluation data according to specified dataset and mode
    dataset_name, mode = args.dataset, args.mode
    data_path = f'results/eval/{dataset_name}-{mode}-semantic.json'
    with open(data_path, 'r') as f:
        data_list = json.load(f)

    # Create LogicalAnalyzer instance
    analyzer = LogicalAnalyzer()

    # Generate SQL query plans
    sql_plan_list = [
        analyzer.get_query_plan(
            data['db_path'], 
            data['pred']
        )
        for data in tqdm(data_list, desc="Generating SQL plans")
    ]

    # # check non-null plans
    # assert all(plan is not None for plan in sql_plan_list)
    filtered = [
        (data, plan)
        for data, plan in zip(data_list, sql_plan_list)
        if plan is not None
    ]
    data_list, sql_plan_list = zip(*filtered)  # 这里会变成tuple，可以 list() 一下
    data_list = list(data_list)
    sql_plan_list = list(sql_plan_list)

    # Construct dataset and save to disk
    save_path = f"data/processed/{dataset_name}-{mode}-plan.pt"
    torch.save(sql_plan_list, save_path)

    data_path = f'data/processed/{dataset_name}-{mode}-eval.json'
    with open(data_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    import dotenv
    dotenv.load_dotenv()

    from src.utils.params import get_args
    args = get_args()

    run(args)