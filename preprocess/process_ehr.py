import json

def load(mode: str):
    with open(f'data/mimic_iv/{mode}/annotated.json') as f:
        data = json.load(f)
    print(mode, len(data))
    return data

if __name__ == '__main__':
    train_data = load('train') 
    valid_data = load('valid') 
    test_data = load('test')

    merge_data = train_data + valid_data
    with open('data/ehr/train/train.json', 'w') as f:
        json.dump(merge_data, f, indent=4)

    merge_data = train_data + valid_data + test_data
    with open('data/ehr/test/test.json', 'w') as f:
        json.dump(merge_data, f, indent=4)