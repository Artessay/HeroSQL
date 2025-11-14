import json
import os

from src.generator import SqlGenerator, load_generate_dataset
from src.utils import seed_everything


def run(args):
    seed_everything(2025)

    # prepare dataset
    dataset_name, mode = args.dataset, args.mode
    data_list = load_generate_dataset(dataset_name, mode)
    
    # generate sql list
    model_name = args.model
    sql_generator = SqlGenerator(model_name, num_samples=64)

    prompt_list = [data['prompt'] for data in data_list]
    responses = sql_generator(prompt_list)

    for data, response in zip(data_list, responses):
        data.pop('prompt')

        # add golden sqls into responses for evaluation
        gt_sql = data['SQL'] if 'SQL' in data else data['query']
        if gt_sql not in response:
            response = [gt_sql] + response
        
        data['sqls'] = response

    model_name = model_name.split('/')[-1]
    save_path = f'results/eval/{dataset_name}-{model_name}-{mode}-raw.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(data_list, f, indent=4)
    
if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()

    from src.utils.params import get_args
    args = get_args()
    args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    run(args)