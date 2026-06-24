#!/bin/bash
#SBATCH -c 32
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/tokenize_domino_mix-%A_%a.out
#SBATCH -e log/error-tokenize_domino_mix-%A_%a.out
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=tokenize_domino_mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

# Tokenize Dolmino Mix 1124 (OLMo2 50B proportions) with datatrove.
#
# Tokenize all sources (one SLURM array task per source):
#   sbatch --array=0-5 scripts/tokenize_domino_mix.sh tokenize
#
# Merge tokenized shards after all array tasks finish:
#   sbatch --dependency=afterok:<JOBID> scripts/tokenize_domino_mix.sh merge
#
# Optional env overrides:
#   DESTINATION=/path/to/data sbatch --array=0-5 scripts/tokenize_domino_mix.sh tokenize
#   TOKENIZER=meta-llama/Llama-2-7b-hf EOS_TOKEN='</s>' ...

set -euo pipefail

STEP="${1:-tokenize}"
DESTINATION="${DESTINATION:-data/dolmino-mix-1124-50B}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-2-7b-hf}"
EOS_TOKEN="${EOS_TOKEN:-</s>}"
SEED="${SEED:-42}"

SOURCES=(dclm flan math pes2o stackexchange wiki)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

mkdir -p log "${DESTINATION}"

source ~/.bashrc
conda activate prm_rlvr

PYTHON="${PYTHON:-python}"

run_tokenize() {
    if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        echo "ERROR: tokenize step requires a SLURM array job (sbatch --array=0-5 ...)" >&2
        exit 1
    fi

    SOURCE="${SOURCES[$SLURM_ARRAY_TASK_ID]}"
    echo "Step=${STEP} source=${SOURCE} destination=${DESTINATION}"
    echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-} SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"

    "${PYTHON}" pretraining/domino_mix_to_tokens.py \
        --step tokenize \
        --source "${SOURCE}" \
        --destination "${DESTINATION}" \
        --tokenizer "${TOKENIZER}" \
        --eos-token "${EOS_TOKEN}" \
        --seed "${SEED}"
}

run_merge() {
    echo "Step=${STEP} destination=${DESTINATION}"

    "${PYTHON}" pretraining/domino_mix_to_tokens.py \
        --step merge \
        --destination "${DESTINATION}" \
        --seed "${SEED}"
}

case "${STEP}" in
    tokenize) run_tokenize ;;
    merge)    run_merge ;;
    *)
        echo "Usage: $0 [tokenize|merge]" >&2
        exit 1
        ;;
esac
