import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
# from .prompt_Cove import build_verify_prompt_2,build_verify_prompt_3,build_verify_prompt_4
from src.baseline.Cove.prompt_Cove import build_verify_prompt_2, build_verify_prompt_3, build_verify_prompt_4

class LLMJudger_Cove:
    def __init__(self, model_path: str):
        # Get the number of available GPUs in the system
        num_gpus = torch.cuda.device_count()

        # Initialize the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
 
        # Initialize the LLM instance from vllm and configure multi-GPU parameters
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.5, tensor_parallel_size=num_gpus)


    # 步骤 2: 规划验证问题
    def _plan_verification_questions(self, questions: list,schemas: list,sqls: list) -> list:
        """根据初始回答生成一系列验证问题。"""
        
        tokenizer = self.tokenizer
        llm = self.llm

        # 1. 在循环外部，先生成验证问题的提示（只执行一次）


        # 3. 在循环内部，对 prompts 列表中的每个 prompt 进行 tokenizer 处理
        texts = []
        for question, schema, sql in zip(questions, schemas, sqls):
            # 在这里同时使用 question, schema, sql
            prompt=build_verify_prompt_2(question, schema, sql)
            
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Set to False to strictly disable thinking
            )
            texts.append(text)

        # 4. 根据你的需求，你可能需要用这个 texts 列表进行后续的批量推理
        
        questions_output = llm.generate(texts, SamplingParams(max_tokens=256, temperature=0.0))
        outputs=[]
        for item in tqdm(questions_output):
            response_content = item.outputs[0].text.strip()
            outputs.append(response_content)
        # 5. 最后返回生成的验证问题列表
        return outputs

    # 步骤  3: 执行验证
    def _execute_verification(self, questions: list,schemas: list,sqls: list, verify_questions: list) -> list:
        """根据初始回答生成一系列验证问题。"""
        
        tokenizer = self.tokenizer
        llm = self.llm

        # 1. 在循环外部，先生成验证问题的提示（只执行一次）


        # 3. 在循环内部，对 prompts 列表中的每个 prompt 进行 tokenizer 处理
        texts = []
        for question, schema, sql, verify_question in zip(questions, schemas, sqls, verify_questions):
            # 在这里同时使用 question, schema, sql
            prompt=build_verify_prompt_3(question, schema, sql, verify_question)
            
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Set to False to strictly disable thinking
            )
            texts.append(text)

        # 4. 根据你的需求，你可能需要用这个 texts 列表进行后续的批量推理
        
        questions_output = llm.generate(texts, SamplingParams(max_tokens=256, temperature=0.0))
        outputs=[]
        for item in tqdm(questions_output):
            response_content = item.outputs[0].text.strip()
            outputs.append(response_content)
        # 5. 最后返回生成的验证问题列表
        return outputs


    #question_list,schema_list,sql_list
    def judge(self, questions: list,schemas: list,sqls: list):
        tokenizer = self.tokenizer
        llm = self.llm
        prompts_2=self._plan_verification_questions(questions,schemas,sqls)
        prompts_3=self._execute_verification(questions,schemas,sqls,prompts_2)
        texts = []
        for question, schema, sql, verify_question,verify_answer in zip(questions, schemas, sqls, prompts_2,prompts_3):
            # 在这里同时使用 question, schema, sql
            prompt=build_verify_prompt_4(question, schema, sql, verify_question,verify_answer)
            
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False, # Set to False to strictly disable thinking
            )
            texts.append(text)
        # Get the token IDs for approve and reject
        approve_token_id = tokenizer.encode("approve", add_special_tokens=False)[0]
        reject_token_id = tokenizer.encode("reject", add_special_tokens=False)[0]

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, seed=42, logprobs=5,
            allowed_token_ids=[approve_token_id, reject_token_id, tokenizer.eos_token_id])
        

        # Perform batch inference using vllm
        # print("texts[0]",texts[0])
        outputs = llm.generate(texts, sampling_params)

        results = []
        for output in tqdm(outputs):
            response_content = output.outputs[0].text.strip()
            # print("response_content",response_content)
            # exit()
            # Get the logits for the first token
            first_token_logprobs = output.outputs[0].logprobs[0]
            
            # Get the approve_logit
            approve_result = first_token_logprobs.get(approve_token_id)
            approve_logit = approve_result.logprob if approve_result else float('-inf')

            # Get the reject_logit
            reject_result = first_token_logprobs.get(reject_token_id)
            reject_logit = reject_result.logprob if reject_result else float('-inf')
            
            # Handle the case where both logits are -inf
            if approve_logit == float('-inf') and reject_logit == float('-inf'):
                approve_logit = float(0)
                reject_logit = float(0)

            # Calculate the approve score
            logits = torch.tensor([approve_logit, reject_logit], dtype=torch.float)
            probabilities = torch.softmax(logits, dim=0)
            approve_score = probabilities[0].item()
            
            item = {
                "output": response_content,
                "score": approve_score,
                "approve_logit": probabilities[0].item(),
                "reject_logit": probabilities[1].item(),
            }

            results.append(item)
        return results

if __name__ == '__main__':
    from src.template import build_verify_prompt

    judger = LLMJudger_Cove("/data1/qrh/model/Qwen3-0.6B/")#"Qwen/Qwen2.5-Coder-7B-Instruct"

    # prompts = [
    #     build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 1"),
    #     build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 2"),
    # ]
    questions=["Find item whose id is 1.", "Find item whose id is 1."]
    schemas=["create table t (id int, name text);", "create table t (id int, name text);"]
    sqls=["select * from t where id = 1","select * from t where id = 1"]
    results = judger.judge(questions,schemas,sqls)
    print(results)