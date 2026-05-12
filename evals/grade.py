from datasets import load_dataset
import argparse
import os
import itertools
import numpy as np
from tqdm import tqdm
import json
from typing import List, Union
from verl.utils.reward_score.simple_math import execute_llm_code, execute_tinygsm_code
from math_verify import parse,verify

def get_llm_answer(text):
    response_type = 'text'
    if '<llm-code>' in text:
        code_out = execute_llm_code(text)
        response_type = 'llm-code'
        if code_out is not None:
            return parse(code_out), 'llm-code'
    if 'def' in text:
        code_out = execute_tinygsm_code(text)
        response_type = 'tinygsm-code'
        if code_out is not None:
            return parse(code_out), 'tinygsm-code'
    
    return parse(text), response_type

def verify_llm_answer(llm_text, answer_text):
    llm_answer, _ = get_llm_answer(llm_text)
    correct_answer = parse(answer_text)
    return verify(llm_answer, correct_answer, timeout_seconds=3)


def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])


def process_jsonl_file(file_name):
    """
    Process a JSONL file and dynamically handle the number of problems.
    """
    results = []
    with open(file_name) as f:
        for line in f:
            data = json.loads(line)
            id = int(data["example_id"])
            while len(results) <= id:  # Ensure the list is large enough
                results.append({"gt": None, "responses": [], "prompt": None})
            gt = data["answer"]
            response = data["response"]
            prompt = data['prompt']
            results[id]["gt"] = gt
            results[id]["prompt"] = prompt
            results[id]["responses"].append(response)
    return results


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OLMo-150M")
    parser.add_argument("--step", type=int)
    parser.add_argument("-b", "--benchmark", type=str, default="Math-500")
    parser.add_argument("--type", type=str, default="constant")
    args = parser.parse_args()

    benchmark_dict = {
        "Math-500": {
            "n": 192,
        },
        "GSM8K": {
            "n": 192,
        },
    }

    if args.step == 0:
        file_path = f"evals/data/{args.model_name}-{args.type}-step60000-pretrain/{args.benchmark.lower()}_t0.6_p1.0_n{benchmark_dict[args.benchmark]['n']}-MNT2048.jsonl"
    else:
        file_path = f"evals/data/{args.model_name}-{args.type}-gsm8k-step60000-grpopp-step_{args.step}/{args.benchmark.lower()}_t0.6_p1.0_n{benchmark_dict[args.benchmark]['n']}-MNT2048.jsonl"
    df = process_jsonl_file(file_path)
    tqdm_loader = tqdm(range(len(df)))
    total = []
    correct = []
    
    for i in tqdm_loader:
        prompt = df[i]['prompt']
        responses = df[i]['responses']
        gt = df[i]['gt']
        all_scores = []

        if i in [65, 188] and args.benchmark == "Math-500":
            all_scores.extend([0] * len(responses))
        else:
            for i, resp in enumerate(responses):
                score = float(verify_llm_answer(resp, gt))
                all_scores.append(score)
        
        total.append(len(responses))
        correct.append(sum(all_scores))
        tqdm_loader.set_postfix(acc=np.mean(all_scores).item())
    
    output_dir = f"eval_results/{args.model_name}"
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, 'pass.jsonl')
    row_data = {
        'model_name': args.model_name + "-grpopp",
        "step": args.step,
        'type': args.type,
        'dataset': args.benchmark,
        'raw_scores': correct,
        'total': total[0],
    }

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    total = np.array(total)
    correct = np.array(correct)
    pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean().item()
                for k in ks if (total >= k).all()}
    print(pass_at_k)
    for k, v in pass_at_k.items():
        row_data[k] = v
    print("JSON path:", json_path)
    # Write to CSV
    with open(json_path, 'a+') as f:
        json.dump(row_data, f)
        f.write('\n')