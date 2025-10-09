import json

if __name__ == '__main__':
    # `train_spider.json` contains 7000 samples, which is the original training dataset
    with open('train_spider.json', 'r') as f:
        train_spider = json.load(f)

    # `train_others.json` contains 1000+ samples, which usually be used as validation dataset
    with open('train_others.json', 'r') as f:
        train_others = json.load(f)

    train = train_spider + train_others

    with open('train.json', 'w') as f:
        json.dump(train, f, indent=4)
