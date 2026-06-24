"""Tokenize Dolmino Mix 1124 with OLMo2 stage-2 50B mixture proportions.

Each source is subsampled according to the official mix table, then tokenized with
the Dolma2 tokenizer used by OLMo2. Run one job per source (recommended on SLURM),
then optionally merge the tokenized shards.

Reference: https://huggingface.co/datasets/allenai/dolmino-mix-1124#mix-compositions
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.readers import HuggingFaceDatasetReader
from datatrove.pipeline.tokens import DocumentTokenizer, DocumentTokenizerMerger

DATASET_NAME = "dolmino-mix-1124-50B"
HF_DATASET = "allenai/dolmino-mix-1124"
TOKENIZER = "meta-llama/Llama-2-7b-hf"
EOS_TOKEN = "</s>"
SEED = 42
TOTAL_MIX_TOKENS = 50_000_000_000

# OLMo2 stage-2 50B mix: fraction of each source corpus to include.
# "mix_pct" is the share of the final ~50B token mixture (for reference).
DOLMINO_50B_SOURCES = {
    "dclm": {"source_pct": 3.23, "mix_pct": 47.2},
    "flan": {"source_pct": 50.0, "mix_pct": 16.6},
    "pes2o": {"source_pct": 5.15, "mix_pct": 5.85},
    "wiki": {"source_pct": 100.0, "mix_pct": 7.11},
    "stackexchange": {"source_pct": 100.0, "mix_pct": 2.45},
    "math": {"source_pct": 100.0, "mix_pct": 20.8},
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
    cfg = DOLMINO_50B_SOURCES[source]
    sample_rate = min(1.0, cfg["source_pct"] / 100.0)
    output_root = os.path.join(destination, f"{DATASET_NAME}-tokenized", source)

    print(
        f"Tokenizing {source}: sample_rate={sample_rate:.4f} "
        f"(target mix share ~{cfg['mix_pct']}% of {TOTAL_MIX_TOKENS / 1e9:.0f}B tokens)"
    )

    reader = HuggingFaceDatasetReader(
        HF_DATASET,
        dataset_options={"name": source, "split": "train"},
        streaming=True,
        text_key="text",
        id_key="id",
    )

    dist_executor = LocalPipelineExecutor(
        pipeline=[
            reader,
            SamplerFilter(rate=sample_rate, seed=seed),
            DocumentTokenizer(
                output_folder=output_root,
                save_filename=source,
                tokenizer_name_or_path=tokenizer,
                eos_token=eos_token,
                shuffle=True,
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

    for source in DOLMINO_50B_SOURCES:
        source_dir = tokenized_root / source
        if not source_dir.exists():
            raise FileNotFoundError(
                f"Missing tokenized output for source '{source}' at {source_dir}. "
                "Run tokenization for all sources first."
            )
        for token_file in source_dir.glob("*.ds"):
            link_path = staging_dir / f"{source}_{token_file.name}"
            if link_path.exists() or link_path.is_symlink():
                link_path.unlink()
            link_path.symlink_to(token_file.resolve())

    # DocumentTokenizerMerger must run as a single process.
    dist_executor = LocalPipelineExecutor(
        pipeline=[
            DocumentTokenizerMerger(
                input_folder=str(staging_dir),
                output_folder=str(merged_dir),
                save_filename=DATASET_NAME,
                shuffle=True,
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
        description="Tokenize Dolmino Mix 1124 using OLMo2 50B mixture proportions."
    )
    parser.add_argument(
        "--step",
        choices=["tokenize", "merge"],
        default="tokenize",
        help="tokenize: subsample and tokenize one source; merge: combine all sources",
    )
    parser.add_argument(
        "--source",
        choices=sorted(DOLMINO_50B_SOURCES.keys()),
        help="Source to tokenize (required for --step tokenize)",
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
