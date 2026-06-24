#!/bin/bash
#SBATCH -c 2 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval_grpo-prolonged-mix.out
#SBATCH -e log/error-eval_grpo-prolonged-mix.out
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_grpo-prolonged-mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -G A100:4

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr

# types=(constant constant-6e4 constant-1e3 constant-3e3)
types=(constant)
# steps=(128 256 384 512 640 768 896 1024)
# steps=(1536 2048 2560 3072)

# steps=(512 1024)
steps=(1024)
rollouts=(16)
lrs=(1e-6)
train_types=(mix)

for type in "${types[@]}";do
for step in "${steps[@]}";do
for train_type in "${train_types[@]}";do
for rollout in "${rollouts[@]}";do
for lr in "${lrs[@]}";do
    # if [[ "$type" == "constant" && "$step" == "1024" ]]; then
    #     continue
    # fi
    python evals/gen_vllm.py --model_path checkpoints/Pretrain-sharpen/OLMo-150M-${type}-${train_type}-step60000-rollout${rollout}-lr${lr}-PPO/global_step_${step}/actor/huggingface \
    --n_gpus 4 \
    --output_path evals/data/OLMo-150M-PPO/OLMo-150M-${type}-${train_type}-step60000-rollout${rollout}-lr${lr}-PPO-step_${step}

#     python evals/gen_vllm.py --model_path OLMo-150M-REINFORCE++/OLMo-150M-${type}-gsm8k-step60000/step_${step} \
#     --n_gpus 4 \
#     --output_path evals/data/OLMo-150M-REINFORCE++/OLMo-150M-${type}-gsm8k-step60000-step_${step}

    # python evals/gen_vllm.py --model_path OLMo-150M/OLMo-150M-${type} \
    # --n_gpus 4 \
    # --output_path evals/data/OLMo-150M/OLMo-150M-${type}-gsm8k-step60000-step_${step}
done
done
done
done
done