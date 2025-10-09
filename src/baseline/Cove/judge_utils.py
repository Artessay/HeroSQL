
from .llm_judger import LLMJudger_Cove


def judge_by_llm(model_path: str, data_list: list):
    judger = LLMJudger_Cove(model_path)
    question_list = [data["question"] for data in data_list]
    schema_list = [data["schema"] for data in data_list]
    sql_list = [data["sql"] for data in data_list]
    # "question": question,
    # "schema": schema,
    # "sql": sql
    results = judger.judge(question_list,schema_list,sql_list)
    scores = [result['score'] for result in results]
    
    return scores
