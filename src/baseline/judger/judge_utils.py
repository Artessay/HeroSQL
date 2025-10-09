
from .llm_judger import LLMJudger


def judge_by_llm(model_path: str, data_list: list):
    judger = LLMJudger(model_path)
    prompt_list = [data['prompt'] for data in data_list]
    results = judger.judge(prompt_list)
    scores = [result['score'] for result in results]
    
    return scores
