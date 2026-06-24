#!/bin/bash
#SBATCH -c 1 # request two cores 
#SBATCH -p kisski-h100,kisski
#SBATCH -o log/convert_hf.out
#SBATCH -e log/error-convert_hf.out
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --job-name=convert_hf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export VLLM_DISABLE_COMPILE_CACHE=1
source ~/.bashrc
conda activate prm_rlvr
# steps=(3000 6000 9000 12000 15000 18000 21000 24000 27000 30000 33000 36000 39000 42000 45000 48000 51000 54000 57000 60000 61777)
# steps=(60000)

# for step in ${steps[@]}; do
#     # python convert_to_hf.py --input_dir OLMo-150M/OLMo-150M-cosine/step${step}-unsharded --output_dir OLMo-150M/OLMo-150M-cosine/step${step}-hf  --no_fix_eos_token_id
#     # python convert_to_hf.py --input_dir OLMo-150M/OLMo-150M-constant/step${step}-unsharded --output_dir OLMo-150M/OLMo-150M-constant/step${step}-hf  --no_fix_eos_token_id

#     python save_tokienizer.py --tokenizer_name meta-llama/Llama-2-7b-hf --target_dir OLMo-150M/OLMo-150M-cosine/step${step}-hf
#     python save_tokienizer.py --tokenizer_name meta-llama/Llama-2-7b-hf --target_dir OLMo-150M/OLMo-150M-constant/step${step}-hf
# done

steps=(128 256 384 512 640 768 896 1024)
steps=(1536 2048 2560 3072)
steps=(1024)
# types=(constant constant-6e4 constant-1e3)
types=(constant)
# types=(cosine)
lrs=(1e-6)

for type in "${types[@]}";do
for step in "${steps[@]}";do
for lr in "${lrs[@]}";do
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/Pretrain-sharpen/OLMo-150M-${type}-mix-step60000-rollout16-lr${lr}-PPO/global_step_${step}/actor\
        --target_dir checkpoints/Pretrain-sharpen/OLMo-150M-${type}-mix-step60000-rollout16-lr${lr}-PPO/global_step_${step}/actor/huggingface
    
done
done
done