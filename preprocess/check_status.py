import concurrent.futures
import json
import os

from func_timeout import FunctionTimedOut
from tqdm import tqdm

from src.utils import seed_everything, SqlExecutor, SqlUtils


def process_data(data):
    db_path = data['db_path']
    gt_sql = data['SQL'] if 'SQL' in data else data['query']

    sql_executor = SqlExecutor(db_path)

    try:
        result = sql_executor.execute_sql(gt_sql)
        if "data" not in result:
            return None
        gt_results = result['data']
    except FunctionTimedOut:
        # print(gt_sql)
        return None  # Skip this entry
    except Exception as e:
        print(e, gt_sql)
        return None  # Skip this entry
    
    answer = []
    sqls = data['sqls']
    for sql in sqls:
        status = ""
        try:
            result = sql_executor.execute_sql(sql)
            if "data" in result:
                sql_result = result["data"]
                is_equal = SqlUtils.is_result_euqal(sql_result, gt_results)
                if is_equal:
                    status = "Success"
                else:
                    status = "Semantic Error"
            else:
                status = "Syntax Error"
        except FunctionTimedOut:
            status = "Runtime Error"
        except RuntimeError as e:
            status = "Plan Error"
        except Exception as e:
            print(e)
            continue
        answer.append({
            "sql": sql,
            "status": status,
        })
    data.pop('sqls')
    data['answer'] = answer
    return data


def run(args):
    seed_everything(2025)

    # prepare dataset
    dataset_name, model_name, mode = args.dataset, args.model, args.mode
    model_name = model_name.split('/')[-1]
    input_path = f'results/eval/{dataset_name}-{model_name}-{mode}-raw.json'
    with open(input_path, 'r') as f:
        data_list = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_data, data_list), total=len(data_list)))
    # Filter out any None (i.e. entries that had FunctionTimedOut on gt_sql)
    save_data_list = [r for r in results if r is not None]

    save_path = f'results/eval/{dataset_name}-{model_name}-{mode}-status.json'
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