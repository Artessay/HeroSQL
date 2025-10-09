import json

from tqdm import tqdm

from src.meta import DBMetaAgent
from src.template import build_verify_prompt, build_verify_prompt_CoT, build_verify_prompt_Confidence

def load_llm_verify_function(method_name: str = "llm"):
    if method_name == 'llm':
        from src.baseline.judger import judge_by_llm
    elif method_name == 'cot':
        from src.baseline.Cotjudger import judge_by_llm
    elif method_name == 'confidence':
        from src.baseline.Confidence import judge_by_llm
    elif method_name == 'cove':
        from src.baseline.Cove import judge_by_llm
    else:
        raise ValueError(f'{method_name} is not supported')
    
    return judge_by_llm

def load_llm_verify_dataset(dataset_name: str, mode: str = 'dev', method: str = "llm"):
    db_meta_agent = DBMetaAgent()
    data_path = f"data/processed/{dataset_name}-{mode}-eval.json"

    with open(data_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    data_list = []
    for item in tqdm(dataset, ncols=80, desc='Loading dataset'):
        # get question
        query = item['question']
        evidence = item.get('evidence', None)
        question = query + "\n" + evidence if evidence else query

        # get schema
        database_path = item['db_path']
        schema = db_meta_agent.get_schema(database_path)

        # get sql and label
        sql = item["pred"]
        #修改这个就可以了
        if method=="llm" or method=="cove":
            prompt = build_verify_prompt(question, schema, sql)
        elif method=="cot":
            prompt = build_verify_prompt_CoT(question, schema, sql)
        elif method=="confidence":
            prompt = build_verify_prompt_Confidence(question, schema, sql)
        else:
            raise ValueError(f'{method} is not supported')
        label = 1 if item["status"] == "Success" else 0
        
        data_list.append({
            "prompt": prompt,
            "label": label,

            "question": question,
            "schema": schema,
            "sql": sql
        })

    return data_list
