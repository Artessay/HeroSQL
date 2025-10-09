import json
import torch

from src.meta import DBMetaAgent

def load_eval_dataset(dataset_name: str, mode: str):
    assert mode in ['train', 'dev']
    data_path = f'data/processed/{dataset_name}-{mode}-eval.json'
    
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data

def load_plan_dataset(dataset_name: str, mode: str):
    assert mode in ['train', 'dev']
    data_path = f'data/processed/{dataset_name}-{mode}-plan.pt'

    return torch.load(data_path, weights_only=False)

def load_triple_stream_dataset(dataset_name: str, mode: str, tight_format: bool = True):
    data_list = load_eval_dataset(dataset_name, mode)
    graph_list = load_plan_dataset(dataset_name, mode)

    # # debug use
    # data_list = data_list[:20]
    # graph_list = graph_list[:20]

    question_list = [
        item['question'] + ' ' + item['evidence']
        if item['evidence'] else item['question'] 
        for item in data_list
    ]
    label_list = [
        1 if item['status'] == 'Success' else 0
        for item in data_list
    ]

    db_meta_agent = DBMetaAgent()
    schema_list = [
        db_meta_agent.get_schema(item['db_path'], tight_format=tight_format)
        for item in data_list
    ]

    return schema_list, question_list, graph_list, label_list

if __name__ == '__main__':
    schema_list, question_list, graph_list = load_triple_stream_dataset('bird', 'dev')
    print(schema_list[0])
    print('-'*10)
    print(question_list[0])
    print('-'*10)
    print(graph_list[0])