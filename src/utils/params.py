
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str, default="llm")
    parser.add_argument('-d', '--dataset', type=str, default='bird')
    parser.add_argument('-l', '--model', type=str, default='Qwen/Qwen3-0.6B')
    parser.add_argument('--mode', type=str, default="dev", choices=["train", "dev"])
    parser.add_argument('-s', '--seed', type=int, default=2025)
    
    args = parser.parse_args()
    print(args)

    return args