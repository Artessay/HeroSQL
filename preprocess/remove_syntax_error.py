import json
import os

from tqdm import tqdm

from src.utils import seed_everything


def make_item(data, item, question_id):
    return {
        'question_id': question_id,
        'db_id': data['db_id'] if 'db_id' in data else 'mimic_iv',
        'question': data['question'],
        'evidence': data['evidence'] if 'evidence' in data else '',
        'difficulty': data['difficulty'] if 'difficulty' in data else '',
        'db_path': data['db_path'],
        'gold': data['SQL'] if 'SQL' in data else data['query'],
        'pred': item['sql'],
        'status': item['status'],
    }

def run(args):
    seed_everything(2025)

    # prepare dataset
    dataset_name, model_name, mode = args.dataset, args.model, args.mode
    model_name = model_name.split('/')[-1]
    input_path = f'results/eval/{dataset_name}-{model_name}-{mode}-status.json'
    with open(input_path, 'r') as f:
        data_list = json.load(f)

    question_id = 0
    save_data_list = []
    for data in tqdm(data_list):
        answer_list = data['answer']
        item_list = []
        for item in answer_list:
            if item['status'] not in ['Success', 'Semantic Error']:
                continue
            item_list.append(make_item(data, item, question_id))
            question_id += 1
        save_data_list.extend(item_list)

    save_path = f'results/eval/{dataset_name}-{mode}-semantic.json'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(save_data_list, f, indent=4)

if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()

    from src.utils.params import get_args
    args = get_args()
    args.model = "Qwen/Qwen2.5-Coder-7B-Instruct"
    run(args)