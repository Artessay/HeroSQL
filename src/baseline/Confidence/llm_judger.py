import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import math

class LLMJudger_Confidence:
    def __init__(self, model_path: str):
        # Get the number of available GPUs in the system
        num_gpus = torch.cuda.device_count()
        self.gamma=0.5
        # Initialize the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
 
        # Initialize the LLM instance from vllm and configure multi-GPU parameters
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.5, tensor_parallel_size=num_gpus)

    def _compute_entropy(self, logprob_dict):
        """
        logprob_dict: {token_id: LogProb object, ...} for one step
        """
        probs = torch.tensor(
            [math.exp(lp.logprob) for lp in logprob_dict.values()],
            dtype=torch.float
        )
        probs = probs / probs.sum()  # normalize if only top-k
        entropy = -(probs * torch.log(probs + 1e-12)).sum().item()
        return entropy


    def judge(self, prompts: list):
        tokenizer = self.tokenizer
        llm = self.llm
        # Get the token IDs for approve and reject
        # approve_token_id = tokenizer.encode("approve", add_special_tokens=False)[0]
        # reject_token_id = tokenizer.encode("reject", add_special_tokens=False)[0]

        sampling_params = SamplingParams(max_tokens=128, temperature=0.0, seed=42, logprobs=5)

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
            generated_text = output.outputs[0].text.strip()
            logprobs_per_step = output.outputs[0].logprobs  # list, each step has dict

            # 计算每个 token 的熵
            entropies = []
            for step in logprobs_per_step:
                entropy = self._compute_entropy(step)
                entropies.append(entropy)

            # 最大熵作为不确定性
            if entropies:
                u = max(entropies)
            else:
                u = float("inf")

            # 转换成置信度分数
            confidence = math.exp(-u)  # 越大越自信
            # print("confidence",confidence)
            # 阈值法判标签
            label = "approve" if confidence >= self.gamma else "reject"

            item = {
                "output": generated_text,
                "score": confidence,
                "uncertainty": u,
                "label": label,
            }
            results.append(item)

        return results

if __name__ == '__main__':
    from src.template import build_verify_prompt

    judger = LLMJudger_Confidence("Qwen/Qwen2.5-Coder-7B-Instruct")

    prompts = [
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 1"),
        build_verify_prompt("Find item whose id is 1.", "create table t (id int, name text);", "select * from t where id = 2"),
    ]
    results = judger.judge(prompts)
    print(results)