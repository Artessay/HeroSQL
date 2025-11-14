import json
from tqdm import tqdm

from src.meta import DBMetaAgent
from src.template import build_generate_prompt

def load_dataset(dataset_name: str, mode: str):
    assert mode in ['train', 'dev']
    data_path = f'data/{dataset_name}/{mode}/{mode}.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_generate_dataset(dataset_name: str, mode: str):
    db_meta_agent = DBMetaAgent()
    dataset = load_dataset(dataset_name, mode)
    
    data_list = []
    for item in tqdm(dataset, ncols=80, desc='Loading dataset'):
        # get question
        query = item['question']
        evidence = item.get('evidence', None)
        
        # get schema
        if dataset_name == 'ehr':
            database_path = 'data/ehr/mimic_iv.sqlite'
        elif dataset_name == 'spider2':
            db_name = item['db_id']
            database_path = f'data/spider2/databases/{db_name}.sqlite'
        else:
            db_name = item['db_id']
            database_path = f'data/{dataset_name}/{mode}/{mode}_databases/{db_name}/{db_name}.sqlite'
        schema = db_meta_agent.get_schema(database_path)
    
        prompt = build_generate_prompt(schema, query, evidence)
        item['db_path'] = database_path
        item['prompt'] = prompt

        data_list.append(item)

    return data_list