import os

from src.baseline.utils import load_llm_verify_dataset, load_llm_verify_function
from src.metrics import evaluate_and_save
from src.utils import seed_everything
#这个是调用的主函数
def run(args):
    seed = args.seed
    seed_everything(seed)

    # prepare args
    model_name = args.model
    method_name = args.method
    dataset_name = args.dataset
    model_base_name = model_name.split('/')[-1]
    save_path = f'results/{dataset_name}/{model_base_name}/{method_name}-{seed}.json'
    # if os.path.exists(save_path):
    #     print(f'{save_path} already exists, skip')
    #     return

    # prepare dataset
    data_list = load_llm_verify_dataset(dataset_name, method=method_name)
    
    # judge by verifier
    judge_by_llm = load_llm_verify_function(method_name)
    score_list = judge_by_llm(model_name, data_list)

    # evaluate
    label_list = [data['label'] for data in data_list]
    evaluate_and_save(score_list, label_list, save_path)
    
if __name__ == '__main__':
    import dotenv
    dotenv.load_dotenv()

    from src.utils.params import get_args
    args = get_args()
    run(args)