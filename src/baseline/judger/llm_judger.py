import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMJudger:
    def __init__(self, model_path: str):
        # Get the number of available GPUs in the system
        num_gpus = torch.cuda.device_count()

        # Initialize the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
 
        # Initialize the LLM instance from vllm and configure multi-GPU parameters
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.8, tensor_parallel_size=num_gpus)

    def judge(self, prompts: list):
        tokenizer = self.tokenizer
        llm = self.llm
        # Get the token IDs for approve and reject
        approve_token_id = tokenizer.encode("approve", add_special_tokens=False)[0]
        reject_token_id = tokenizer.encode("reject", add_special_tokens=False)[0]

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, seed=42, logprobs=5,
            allowed_token_ids=[approve_token_id, reject_token_id, tokenizer.eos_token_id])

        # Prepare all input prompts
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Set to False to strictly disable thinking
            )
            texts.append(text)

        # Perform batch inference using vllm
        outputs = llm.generate(texts, sampling_params)

        results = []
        for output in tqdm(outputs):
            response_content = output.outputs[0].text.strip()
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

    judger = LLMJudger("Qwen/Qwen2.5-Coder-7B-Instruct")

    prompts = [
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 1"),
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 2"),
    ]
    results = judger.judge(prompts)
    print(results)