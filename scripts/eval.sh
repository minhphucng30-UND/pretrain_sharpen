#!/bin/bash
#SBATCH -c 8 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/eval_grpo-rollout.out
#SBATCH -e log/error-eval_grpo-rollout.out
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --job-name=eval_grpo-rollout
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -G H100:4

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr

# types=(constant constant-6e4 constant-1e3 constant-3e3)
types=(constant-6e4 constant-1e3 constant-3e3 cosine)
steps=(128 256 384 512 640 768 896 1024)
steps=(1024)
rollouts=(16)

for type in "${types[@]}";do
for step in "${steps[@]}";do
for rollout in "${rollouts[@]}";do
# model_name=OLMo-150M-${type}
    python evals/gen_vllm.py --model_path checkpoints/Pretrain-sharpen/OLMo-150M-${type}-mix-step60000-rollout${rollout}-lr1e-6/global_step_${step}/actor/huggingface \
    --n_gpus 4 \
    --output_path evals/data/OLMo-150M-REINFORCE++/OLMo-150M-${type}-mix-step60000-rollout${rollout}-step_${step}

#     python evals/gen_vllm.py --model_path OLMo-150M-REINFORCE++/OLMo-150M-${type}-gsm8k-step60000/step_${step} \
#     --n_gpus 4 \
#     --output_path evals/data/OLMo-150M-REINFORCE++/OLMo-150M-${type}-gsm8k-step60000-step_${step}

    # python evals/gen_vllm.py --model_path OLMo-150M/OLMo-150M-${type} \
    # --n_gpus 4 \
    # --output_path evals/data/OLMo-150M/OLMo-150M-${type}-gsm8k-step60000-step_${step}
done

done
done