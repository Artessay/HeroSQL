import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class LLMGenerator:
    def __init__(self, model_path: str, num_samples: int = 10):
        # Get the number of available GPUs in the system
        num_gpus = torch.cuda.device_count()

        # Initialize the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Initialize the LLM instance from vllm and configure multi-GPU parameters
        self.llm = LLM(model=model_path, gpu_memory_utilization=0.9, tensor_parallel_size=num_gpus)

        # Initialize the sampling parameters
        self.sampling_params = SamplingParams(
            n=num_samples, 
            max_tokens=512,
            seed=42,
            temperature=0.2,
            top_p=0.9,
            top_k=50,
        )


    def generate(self, prompts: list):
        llm = self.llm
        tokenizer = self.tokenizer
        sampling_params = self.sampling_params

        # Prepare all input prompts
        texts = []
        for prompt in prompts:
            text = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Perform batch inference using vllm
        outputs = llm.generate(texts, sampling_params)

        results = []
        for output in outputs:
            response_contents = [
                item.text.strip() for item in output.outputs
            ]

            results.append(response_contents)
            
        return results

if __name__ == '__main__':
    # prompts = [
    #     [{"role": "user", "content": "What is the capital of France?"}],
    #     [{"role": "user", "content": "How many people live in London?"}],
    # ]
    prompts = [
        [
            {
                "role": "system",
                "content": "As a data analysis expert, you are to extract the necessary information from the data provided and output the corresponding SQL query based on the user's question."
            },
            {
                "role": "user",
                "content": "The database schema is as follows:\nCREATE TABLE satscores (\n\tcds\tTEXT\tNOT NULL\tPRIMARY KEY,\n\trtype\tTEXT\tNOT NULL,\n\tsname\tTEXT\tNULL,\n\tdname\tTEXT\tNULL,\n\tcname\tTEXT\tNULL,\n\tenroll12\tINTEGER\tNOT NULL,\n\tNumTstTakr\tINTEGER\tNOT NULL,\n\tAvgScrRead\tINTEGER\tNULL,\n\tAvgScrMath\tINTEGER\tNULL,\n\tAvgScrWrite\tINTEGER\tNULL,\n\tNumGE1500\tINTEGER\tNULL,\n\tforeign key (cds) references schools (CDSCode)\n);\n\nCREATE TABLE schools (\n\tCDSCode\tTEXT\tNOT NULL\tPRIMARY KEY,\n\tNCESDist\tTEXT\tNULL,\n\tNCESSchool\tTEXT\tNULL,\n\tStatusType\tTEXT\tNOT NULL,\n\tCounty\tTEXT\tNOT NULL,\n\tDistrict\tTEXT\tNOT NULL,\n\tSchool\tTEXT\tNULL,\n\tStreet\tTEXT\tNULL,\n\tStreetAbr\tTEXT\tNULL,\n\tCity\tTEXT\tNULL,\n\tZip\tTEXT\tNULL,\n\tState\tTEXT\tNULL,\n\tMailStreet\tTEXT\tNULL,\n\tMailStrAbr\tTEXT\tNULL,\n\tMailCity\tTEXT\tNULL,\n\tMailZip\tTEXT\tNULL,\n\tMailState\tTEXT\tNULL,\n\tPhone\tTEXT\tNULL,\n\tExt\tTEXT\tNULL,\n\tWebsite\tTEXT\tNULL,\n\tOpenDate\tDATE\tNULL,\n\tClosedDate\tDATE\tNULL,\n\tCharter\tINTEGER\tNULL,\n\tCharterNum\tTEXT\tNULL,\n\tFundingType\tTEXT\tNULL,\n\tDOC\tTEXT\tNOT NULL,\n\tDOCType\tTEXT\tNOT NULL,\n\tSOC\tTEXT\tNULL,\n\tSOCType\tTEXT\tNULL,\n\tEdOpsCode\tTEXT\tNULL,\n\tEdOpsName\tTEXT\tNULL,\n\tEILCode\tTEXT\tNULL,\n\tEILName\tTEXT\tNULL,\n\tGSoffered\tTEXT\tNULL,\n\tGSserved\tTEXT\tNULL,\n\tVirtual\tTEXT\tNULL,\n\tMagnet\tINTEGER\tNULL,\n\tLatitude\tREAL\tNULL,\n\tLongitude\tREAL\tNULL,\n\tAdmFName1\tTEXT\tNULL,\n\tAdmLName1\tTEXT\tNULL,\n\tAdmEmail1\tTEXT\tNULL,\n\tAdmFName2\tTEXT\tNULL,\n\tAdmLName2\tTEXT\tNULL,\n\tAdmEmail2\tTEXT\tNULL,\n\tAdmFName3\tTEXT\tNULL,\n\tAdmLName3\tTEXT\tNULL,\n\tAdmEmail3\tTEXT\tNULL,\n\tLastUpdate\tDATE\tNOT NULL\n);\n\nCREATE TABLE frpm (\n\tCDSCode\tTEXT\tNOT NULL\tPRIMARY KEY,\n\t`Academic Year`\tTEXT\tNULL,\n\t`County Code`\tTEXT\tNULL,\n\t`District Code`\tINTEGER\tNULL,\n\t`School Code`\tTEXT\tNULL,\n\t`County Name`\tTEXT\tNULL,\n\t`District Name`\tTEXT\tNULL,\n\t`School Name`\tTEXT\tNULL,\n\t`District Type`\tTEXT\tNULL,\n\t`School Type`\tTEXT\tNULL,\n\t`Educational Option Type`\tTEXT\tNULL,\n\t`NSLP Provision Status`\tTEXT\tNULL,\n\t`Charter School (Y/N)`\tINTEGER\tNULL,\n\t`Charter School Number`\tTEXT\tNULL,\n\t`Charter Funding Type`\tTEXT\tNULL,\n\tIRC\tINTEGER\tNULL,\n\t`Low Grade`\tTEXT\tNULL,\n\t`High Grade`\tTEXT\tNULL,\n\t`Enrollment (K-12)`\tREAL\tNULL,\n\t`Free Meal Count (K-12)`\tREAL\tNULL,\n\t`Percent (%) Eligible Free (K-12)`\tREAL\tNULL,\n\t`FRPM Count (K-12)`\tREAL\tNULL,\n\t`Percent (%) Eligible FRPM (K-12)`\tREAL\tNULL,\n\t`Enrollment (Ages 5-17)`\tREAL\tNULL,\n\t`Free Meal Count (Ages 5-17)`\tREAL\tNULL,\n\t`Percent (%) Eligible Free (Ages 5-17)`\tREAL\tNULL,\n\t`FRPM Count (Ages 5-17)`\tREAL\tNULL,\n\t`Percent (%) Eligible FRPM (Ages 5-17)`\tREAL\tNULL,\n\t`2013-14 CALPADS Fall 1 Certification Status`\tINTEGER\tNULL,\n\tforeign key (CDSCode) references schools (CDSCode)\n);\n\nWrite SQL for the following question: What is the highest eligible free rate for K-12 students in the schools in Alameda County?\nConsider the extra knowledge, it is very useful to help you understand the question and the corresponding SQL: Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`\nAnd lastly, only write SQL with no comments."
            }
        ],
        [
            {
                "role": "system",
                "content": "As a data analysis expert, you are to extract the necessary information from the data provided and output the corresponding SQL query based on the user's question."
            },
            {
                "role": "user",
                "content": "The database schema is as follows:\nCREATE TABLE satscores (\n\tcds\tTEXT\tNOT NULL\tPRIMARY KEY,\n\trtype\tTEXT\tNOT NULL,\n\tsname\tTEXT\tNULL,\n\tdname\tTEXT\tNULL,\n\tcname\tTEXT\tNULL,\n\tenroll12\tINTEGER\tNOT NULL,\n\tNumTstTakr\tINTEGER\tNOT NULL,\n\tAvgScrRead\tINTEGER\tNULL,\n\tAvgScrMath\tINTEGER\tNULL,\n\tAvgScrWrite\tINTEGER\tNULL,\n\tNumGE1500\tINTEGER\tNULL,\n\tforeign key (cds) references schools (CDSCode)\n);\n\nCREATE TABLE schools (\n\tCDSCode\tTEXT\tNOT NULL\tPRIMARY KEY,\n\tNCESDist\tTEXT\tNULL,\n\tNCESSchool\tTEXT\tNULL,\n\tStatusType\tTEXT\tNOT NULL,\n\tCounty\tTEXT\tNOT NULL,\n\tDistrict\tTEXT\tNOT NULL,\n\tSchool\tTEXT\tNULL,\n\tStreet\tTEXT\tNULL,\n\tStreetAbr\tTEXT\tNULL,\n\tCity\tTEXT\tNULL,\n\tZip\tTEXT\tNULL,\n\tState\tTEXT\tNULL,\n\tMailStreet\tTEXT\tNULL,\n\tMailStrAbr\tTEXT\tNULL,\n\tMailCity\tTEXT\tNULL,\n\tMailZip\tTEXT\tNULL,\n\tMailState\tTEXT\tNULL,\n\tPhone\tTEXT\tNULL,\n\tExt\tTEXT\tNULL,\n\tWebsite\tTEXT\tNULL,\n\tOpenDate\tDATE\tNULL,\n\tClosedDate\tDATE\tNULL,\n\tCharter\tINTEGER\tNULL,\n\tCharterNum\tTEXT\tNULL,\n\tFundingType\tTEXT\tNULL,\n\tDOC\tTEXT\tNOT NULL,\n\tDOCType\tTEXT\tNOT NULL,\n\tSOC\tTEXT\tNULL,\n\tSOCType\tTEXT\tNULL,\n\tEdOpsCode\tTEXT\tNULL,\n\tEdOpsName\tTEXT\tNULL,\n\tEILCode\tTEXT\tNULL,\n\tEILName\tTEXT\tNULL,\n\tGSoffered\tTEXT\tNULL,\n\tGSserved\tTEXT\tNULL,\n\tVirtual\tTEXT\tNULL,\n\tMagnet\tINTEGER\tNULL,\n\tLatitude\tREAL\tNULL,\n\tLongitude\tREAL\tNULL,\n\tAdmFName1\tTEXT\tNULL,\n\tAdmLName1\tTEXT\tNULL,\n\tAdmEmail1\tTEXT\tNULL,\n\tAdmFName2\tTEXT\tNULL,\n\tAdmLName2\tTEXT\tNULL,\n\tAdmEmail2\tTEXT\tNULL,\n\tAdmFName3\tTEXT\tNULL,\n\tAdmLName3\tTEXT\tNULL,\n\tAdmEmail3\tTEXT\tNULL,\n\tLastUpdate\tDATE\tNOT NULL\n);\n\nCREATE TABLE frpm (\n\tCDSCode\tTEXT\tNOT NULL\tPRIMARY KEY,\n\t`Academic Year`\tTEXT\tNULL,\n\t`County Code`\tTEXT\tNULL,\n\t`District Code`\tINTEGER\tNULL,\n\t`School Code`\tTEXT\tNULL,\n\t`County Name`\tTEXT\tNULL,\n\t`District Name`\tTEXT\tNULL,\n\t`School Name`\tTEXT\tNULL,\n\t`District Type`\tTEXT\tNULL,\n\t`School Type`\tTEXT\tNULL,\n\t`Educational Option Type`\tTEXT\tNULL,\n\t`NSLP Provision Status`\tTEXT\tNULL,\n\t`Charter School (Y/N)`\tINTEGER\tNULL,\n\t`Charter School Number`\tTEXT\tNULL,\n\t`Charter Funding Type`\tTEXT\tNULL,\n\tIRC\tINTEGER\tNULL,\n\t`Low Grade`\tTEXT\tNULL,\n\t`High Grade`\tTEXT\tNULL,\n\t`Enrollment (K-12)`\tREAL\tNULL,\n\t`Free Meal Count (K-12)`\tREAL\tNULL,\n\t`Percent (%) Eligible Free (K-12)`\tREAL\tNULL,\n\t`FRPM Count (K-12)`\tREAL\tNULL,\n\t`Percent (%) Eligible FRPM (K-12)`\tREAL\tNULL,\n\t`Enrollment (Ages 5-17)`\tREAL\tNULL,\n\t`Free Meal Count (Ages 5-17)`\tREAL\tNULL,\n\t`Percent (%) Eligible Free (Ages 5-17)`\tREAL\tNULL,\n\t`FRPM Count (Ages 5-17)`\tREAL\tNULL,\n\t`Percent (%) Eligible FRPM (Ages 5-17)`\tREAL\tNULL,\n\t`2013-14 CALPADS Fall 1 Certification Status`\tINTEGER\tNULL,\n\tforeign key (CDSCode) references schools (CDSCode)\n);\n\nWrite SQL for the following question: Please list the lowest three eligible free rates for students aged 5-17 in continuation schools.\nConsider the extra knowledge, it is very useful to help you understand the question and the corresponding SQL: Eligible free rates for students aged 5-17 = `Free Meal Count (Ages 5-17)` / `Enrollment (Ages 5-17)`\nAnd lastly, only write SQL with no comments."
            }
        ]
    ]
    generator = LLMGenerator("Qwen/Qwen2.5-Coder-7B-Instruct", num_samples=4)
    results = generator.generate(prompts)
    for i, result in enumerate(results):
        print(f"\nPrompt {i}:")
        for j, ans in enumerate(result):
            print(f"  Answer {j+1}: {ans}")