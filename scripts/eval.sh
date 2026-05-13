export VLLM_DISABLE_COMPILE_CACHE=1
types=(constant-6e4)
steps=(1024)

for type in "${types[@]}";do
for step in "${steps[@]}";do
model_name=OLMo-150M-${type}
    python evals/gen_vllm.py --model_path checkpoints/Pretrain-sharpen/OLMo-150M-${type}-gsm8k-step60000-6e4/global_step_${step}/actor/huggingface \
     --n_gpus 2 \
     --output_path evals/data/OLMo-150M-${type}-gsm8k-step60000-step_${step}
    # python evals/gen_vllm.py --model_path OLMo-150M/${model_name}/step60000-hf --n_gpus 4 --output_path evals/data/OLMo-150M-${type}-step60000-pretrain
done
done