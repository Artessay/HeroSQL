
from .llm_judger import LLMJudger_Confidence


def judge_by_llm(model_path: str, data_list: list):
    judger = LLMJudger_Confidence(model_path)
    prompt_list = [data['prompt'] for data in data_list]
    results = judger.judge(prompt_list)
    scores = [result['score'] for result in results]
    
    return scores
