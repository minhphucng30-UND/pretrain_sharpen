#!/bin/bash
#SBATCH -c 4 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/grade_grpo-prolonged-mix.out
#SBATCH -e log/error-grade_grpo-prolonged-mix.out
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --job-name=grade_grpo-prolonged-mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

source ~/.bashrc
conda activate prm_rlvr
# steps=(128 256 384 512 640 768 896)
# steps=(1536 2048 2560 3072)
steps=(512 1024 1536 2048 2560 3072)
# steps=(512)
steps=(1024)
# types=(constant constant-6e4 constant-1e3)
types=(constant)
# types=(constant-3e3)

for type in "${types[@]}"; do
for step in "${steps[@]}"; do
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark GSM8K --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark Math-500 --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark Olympiad-Bench --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark AMC23 --train_type mix --rollout_n 16

done
done