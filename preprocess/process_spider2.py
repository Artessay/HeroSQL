import os
import json

def transform(data: dict) -> dict:
    # get gold sql
    gold_path = f'data/spider2/gold/{data["instance_id"]}.sql'
    if not os.path.exists(gold_path):
        # print(f"{gold_path} does not exist")
        return None
    with open(gold_path, 'r') as f:
        gold_sql = f.read().strip()

    # get evidence
    evidence = ""
    if data["external_knowledge"]:
        document_path = f"data/spider2/documents/{data["external_knowledge"]}"
        assert os.path.exists(document_path)

        with open(document_path, "r") as f:
            evidence = f.read()
    
    item = {
        "db_id": data["db"],
        "question": data["question"],
        "evidence": evidence,
        "SQL": gold_sql
    }
    return item

def main():
    with open("data/spider2/spider2-lite.jsonl", "r") as f:
        data = [json.loads(line) for line in f]
    
    # only retain examples using local databases
    data = [transform(d) for d in data if d['instance_id'].startswith('local')]
    print(len(data))

    data = [d for d in data if d]  # remove None items
    print(len(data))

    with open("data/spider2/dev/dev.json", "w") as f:
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()