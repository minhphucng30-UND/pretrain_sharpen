from transformers import AutoTokenizer
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--target_dir", type=str, default="pretrain_data/tokenizer")
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.save_pretrained(args.target_dir)