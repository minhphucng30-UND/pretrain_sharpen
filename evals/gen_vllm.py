import os
import json
import random
import concurrent.futures
import multiprocessing
from pathlib import Path
import argparse
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
#                   Global constants / variables                              #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#                               Helper functions                              #
# --------------------------------------------------------------------------- #
def load_samples(filepath: str):
    """Read parquet file and return a list of prompts (no duplication)."""
    df = pd.read_parquet(filepath)
    if "BRUMO25" in filepath or "CMIMC25" in filepath or "HMMT25" in filepath:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "problem"].strip(),
                "answer": df.at[i, "answer"].strip(),
            }
            for i in range(len(df))
        ]
    else:
        samples = [
            {
                "example_id": i,
                "prompt": df.at[i, "prompt"][0]["content"].strip(),
                "answer": df.at[i, "reward_model"]["ground_truth"].strip(),
            }
            for i in range(len(df))
        ]
    print(f"Total unique samples: {len(samples)}")
    return samples


def split_seeds(seeds: list[int], num_workers: int):
    """Round-robin split of the seed list into num_workers chunks."""
    chunks = [[] for _ in range(num_workers)]
    for idx, s in enumerate(seeds):
        chunks[idx % num_workers].append(s)
    return chunks


# --------------------------------------------------------------------------- #
#                           Worker process (one GPU)                          #
# --------------------------------------------------------------------------- #
def worker_process(args_tuple):
    """
    Each worker runs on a single GPU.

    args_tuple = (samples, seed_list, gpu_id, model_path, temperature, top_p, max_tokens)
    """
    samples, seed_list, gpu_id, model_path, temperature, top_p, max_tokens = args_tuple
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # vLLM forks engine workers; CUDA must not follow fork after parent init — use spawn.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from vllm import LLM, SamplingParams

    print(f"[GPU {gpu_id}] seeds={seed_list} | loading model...", flush=True)

    llm = LLM(
        model=model_path,
        enforce_eager=False,
        max_num_seqs=2048,
    )
    results = []

    for seed in seed_list:
        sampling = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
        prompts = [s["prompt"] for s in samples]
        outputs = llm.generate(prompts, sampling_params=sampling, use_tqdm=True)
        for sample, out in zip(samples, outputs):
            results.append(
                {
                    "example_id": sample["example_id"],
                    "prompt": sample["prompt"],
                    "answer": sample["answer"],
                    "seed": seed,
                    "response": out.outputs[0].text,
                }
            )
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="evals/data")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--n_gpus", type=int, default=4)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    DATA_DIR = args.data_dir
    TASKS       = [
        # {"name": "Math", "path": f"{DATA_DIR}/math/train.parquet", "N": 32},
        # {"name": "Polaris", "path": f"{DATA_DIR}/polaris/train.parquet", "N": 16},
        {"name": "MATH-500", "path": f"{DATA_DIR}/MATH-500/test.parquet", "N": 192},
        {"name": "GSM8K", "path": f"data/gsm8k/test.parquet", "N": 192},
        # {"name": "AIME24", "path": f"{DATA_DIR}/AIME24/test.parquet", "N": 1280},
        # {"name": "AIME25", "path": f"{DATA_DIR}/AIME25/test.parquet", "N": 1280},
    ]

    PROMPT_TEMPLATE = """{problem}"""
    # PROMPT_TEMPLATE = """{problem}"""
    NAME = args.model_path
    MAX_TOKENS  = 2048
    TEMPERATURE = args.temperature
    TOP_P       = 1.0
    OUT_DIR = Path(args.output_path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    available_workers = range(args.n_gpus)
    num_workers = len(available_workers)
    for task in TASKS:
        task_name = task["name"]
        task_path = task["path"]
        N = task["N"]

        print(f"Starting evaluation for task: {task_name} (N={N})")

        # Update output path for the current task
        out_path = OUT_DIR / f"{task_name.lower()}_t{TEMPERATURE}_p{TOP_P}_n{N}-MNT{MAX_TOKENS}.jsonl"
        if os.path.exists(out_path):
            print(f"File {out_path} exists, skipping...")
            continue
        
        # 1. Load original prompts
        samples = load_samples(task_path)

        # Append suffix prompt to each sample
        for sample in samples:
            sample["prompt"] = PROMPT_TEMPLATE.format(problem=sample["prompt"])

        # demo print
        print("Example prompt after formatting:")
        print(samples[0]["prompt"])
        
        # 2. Generate N distinct random seeds and split across GPUs
        random_seeds = random.sample(range(2**31 - 1), N)  # unique & shuffled
        seed_chunks = split_seeds(random_seeds, num_workers)

        # 3. Launch workers
        all_results = []
        args_list = [
            (
                samples,
                seed_chunks[i],
                gid,
                NAME,
                TEMPERATURE,
                TOP_P,
                MAX_TOKENS,
            )
            for (i, gid) in enumerate(available_workers)
        ]
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=ctx,
        ) as ex:
            futures = [ex.submit(worker_process, tup) for tup in args_list]
            for fut in tqdm(concurrent.futures.as_completed(futures),
                            total=len(futures), desc=f"GPU workers ({task_name})"):
                all_results.extend(fut.result())

        print(f"Total generations collected for {task_name}: {len(all_results)}")  # len(samples) * N

        # 4. Save to disk
        with out_path.open("w", encoding="utf-8") as f:
            for item in all_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved results for {task_name} to {out_path}")