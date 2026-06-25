"""Tokenize the math subset of Dolma 3 Dolmino Mix to ~10B tokens.

Four of the official math (synth) sources are included (Dolmino Math is
excluded). Each is subsampled uniformly to preserve their relative proportions
within the math subset.

Run one job per source (recommended on SLURM), then optionally merge the shards.

Reference: https://huggingface.co/datasets/allenai/dolma3_dolmino_mix-100B-1025
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.tokens import DocumentTokenizer, DocumentTokenizerMerger

DATASET_NAME = "dolma3-dolmino-math-10B"
HF_PATH = "hf://datasets/allenai/dolma3_dolmino_mix-100B-1025"
TOKENIZER = "meta-llama/Llama-2-7b-hf"
EOS_TOKEN = "</s>"
SEED = 42
TOTAL_MIX_TOKENS = 10_000_000_000

# Math (synth) sources from the official 100B mix.
# token_count values are from the dataset card (used to compute subsample rate).
# hf_dir is the actual directory name under data/ in the HF repo (note hyphens).
DOLMA3_MATH_SOURCES = {
    "cranemath": {"hf_dir": "cranemath", "mix_pct": 5.63, "token_count": 5_620_000_000},
    "megamatt": {"hf_dir": "megamatt", "mix_pct": 1.73, "token_count": 1_730_000_000},
    "tinyMATH_mind": {"hf_dir": "tinyMATH-mind", "mix_pct": 0.9, "token_count": 898_000_000},
    "tinyMATH_pot": {"hf_dir": "tinyMATH-pot", "mix_pct": 0.24, "token_count": 241_000_000},
}

MATH_SUBSET_TOKENS = sum(cfg["token_count"] for cfg in DOLMA3_MATH_SOURCES.values())
SAMPLE_RATE = min(1.0, TOTAL_MIX_TOKENS / MATH_SUBSET_TOKENS)


def default_adapter(self, data: dict, path: str, id_in_file: int | str):
    return {
        "text": data.pop(self.text_key, ""),
        "id": data.pop(self.id_key, f"{path}/{id_in_file}"),
        "media": [],
        "metadata": {},
    }


def _slurm_task_layout() -> tuple[int, int, int]:
    n_tasks_per_node = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    n_nodes = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    rank = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    return n_tasks_per_node, n_nodes, rank


def _tokenize_source(
    source: str,
    destination: str,
    tokenizer: str,
    eos_token: str,
    seed: int,
    n_tasks_per_node: int,
    n_nodes: int,
    rank: int,
) -> None:
    cfg = DOLMA3_MATH_SOURCES[source]
    output_root = os.path.join(destination, f"{DATASET_NAME}-tokenized", source)

    print(
        f"Tokenizing {source}: sample_rate={SAMPLE_RATE:.4f} "
        f"(100B math share: {cfg['mix_pct']}%, ~{cfg['token_count'] * SAMPLE_RATE / 1e9:.2f}B tokens)"
    )

    reader = JsonlReader(
        HF_PATH,  # read directly from huggingface
        glob_pattern=f"data/{cfg['hf_dir']}/*.jsonl.zst",
        compression="zstd",
        text_key="text",
        id_key="id",
        adapter=default_adapter,
    )

    dist_executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            SamplerFilter(rate=SAMPLE_RATE, seed=seed),
            DocumentTokenizer(
                output_folder=output_root,
                save_filename=source,
                tokenizer_name_or_path=tokenizer,
                eos_token=eos_token,
                shuffle_documents=True,
                seed=seed,
            ),
        ],
        tasks=n_tasks_per_node * n_nodes,
        workers=-1,
        logging_dir=os.path.join(destination, "logs", "datatrove", DATASET_NAME, source),
        local_tasks=n_tasks_per_node,
        local_rank_offset=rank * n_tasks_per_node,
        start_method="fork",
    )
    dist_executor.run()


def _merge_tokenized_sources(destination: str, seed: int) -> None:
    tokenized_root = Path(destination) / f"{DATASET_NAME}-tokenized"
    staging_dir = tokenized_root / "_merge_staging"
    merged_dir = tokenized_root / "merged"

    if staging_dir.exists():
        for path in staging_dir.iterdir():
            path.unlink()
    else:
        staging_dir.mkdir(parents=True, exist_ok=True)
    merged_dir.mkdir(parents=True, exist_ok=True)

    for source in DOLMA3_MATH_SOURCES:
        source_dir = tokenized_root / source
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Missing tokenized output for source '{source}' at {source_dir}. "
                "Run tokenization for all math sources first."
            )
        for token_file in source_dir.glob("*.ds"):
            link_path = staging_dir / f"{source}_{token_file.name}"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(token_file.resolve())

    dist_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=str(staging_dir),
                output_folder=str(merged_dir),
                save_filename=DATASET_NAME,
                shuffle_documents=True,
                seed=seed,
            ),
        ],
        tasks=1,
        workers=1,
        logging_dir=os.path.join(destination, "logs", "datatrove", DATASET_NAME, "merge"),
        start_method="fork",
    )
    dist_executor.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize the math subset of Dolma 3 Dolmino Mix to ~10B tokens."
    )
    parser.add_argument(
        "--step",
        choices=["tokenize", "merge"],
        default="tokenize",
        help="tokenize: subsample and tokenize one source; merge: combine all sources",
    )
    parser.add_argument(
        "--source",
        choices=sorted(DOLMA3_MATH_SOURCES.keys()),
        help="Math HF config to tokenize (required for --step tokenize)",
    )
    parser.add_argument(
        "--destination",
        required=True,
        help="Root directory for tokenized output and logs",
    )
    parser.add_argument("--tokenizer", default=TOKENIZER)
    parser.add_argument("--eos-token", default=EOS_TOKEN)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    n_tasks_per_node, n_nodes, rank = _slurm_task_layout()
    print(
        f"Running with {n_tasks_per_node} tasks per node, {n_nodes} nodes, and rank {rank}"
    )
    print(
        f"Math subset: {MATH_SUBSET_TOKENS / 1e9:.2f}B tokens -> "
        f"{TOTAL_MIX_TOKENS / 1e9:.0f}B at sample_rate={SAMPLE_RATE:.4f}"
    )

    if args.step == "tokenize":
        if args.source is None:
            parser.error("--source is required when --step tokenize")
        _tokenize_source(
            source=args.source,
            destination=args.destination,
            tokenizer=args.tokenizer,
            eos_token=args.eos_token,
            seed=args.seed,
            n_tasks_per_node=n_tasks_per_node,
            n_nodes=n_nodes,
            rank=rank,
        )
    else:
        _merge_tokenized_sources(destination=args.destination, seed=args.seed)


if __name__ == "__main__":
    main()
