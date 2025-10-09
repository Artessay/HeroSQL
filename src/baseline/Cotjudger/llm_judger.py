import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMJudgerCoT:
    def __init__(self, model_path: str):
        # Get the number of available GPUs in the system
        num_gpus = torch.cuda.device_count()

        # Initialize the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
 
        # Initialize the LLM instance from vllm and configure multi-GPU parameters
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.5, tensor_parallel_size=num_gpus)

    def judge(self, prompts: list):
        tokenizer = self.tokenizer
        llm = self.llm
        
        # Get the token IDs for approve and reject
        approve_token_id = tokenizer.encode("approve", add_special_tokens=False)[0]
        reject_token_id = tokenizer.encode("reject", add_special_tokens=False)[0]

        sampling_params = SamplingParams(
            max_tokens=512, 
            temperature=0.0, 
            seed=42, 
            logprobs=5  # collect logprobs
        )

        texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
        
        outputs = llm.generate(texts, sampling_params)
        results = []
        for output in tqdm(outputs):
            response_content = output.outputs[0].text.strip()
            response_tokens = output.outputs[0].token_ids
            response_logprobs = output.outputs[0].logprobs

            if len(response_tokens) >= 2:
                last_token_id = response_tokens[-2]
                last_token_logprobs = response_logprobs[-2]
            else:
                # 兜底逻辑，用最后一个 token
                last_token_id = response_tokens[-1] if response_tokens else None
                last_token_logprobs = response_logprobs[-1] if response_logprobs else None

            approve_result = last_token_logprobs.get(approve_token_id)
            approve_logit = approve_result.logprob if approve_result else float('-inf')

            reject_result = last_token_logprobs.get(reject_token_id)
            reject_logit = reject_result.logprob if reject_result else float('-inf')
            if approve_logit==float('-inf') and reject_logit==float('-inf'):
                approve_score=0.5
            else:
                logits = torch.tensor([approve_logit, reject_logit], dtype=torch.float)
                probabilities = torch.softmax(logits, dim=0)
                approve_score = probabilities[0].item()

            # final judgment from last token text
            final_text = tokenizer.decode([last_token_id]).strip()
            # print("response_content",response_content)
            # print("final_text",final_text)
            # exit()
            if final_text == "approve":
                final_judgment = "approve"
            elif final_text == "reject":
                final_judgment = "reject"
            else:
                final_judgment = "unknown"

            item = {
                "output": response_content,
                "final_judgment": final_judgment,
                "score": approve_score,
                "approve_logit": approve_logit,
                "reject_logit": reject_logit,
            }

            results.append(item)
        return results


if __name__ == '__main__':
    from src.template import build_verify_prompt

    judger = LLMJudgerCoT("Qwen/Qwen2.5-Coder-7B-Instruct")

    prompts = [
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 1"),
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 2"),
    ]
    results = judger.judge(prompts)
    print(results)