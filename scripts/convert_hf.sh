# steps=(3000 6000 9000 12000 15000 18000 21000 24000 27000 30000 33000 36000 39000 42000 45000 48000 51000 54000 57000 60000 61777)
# steps=(60000)

# for step in ${steps[@]}; do
#     # python convert_to_hf.py --input_dir OLMo-150M/OLMo-150M-cosine/step${step}-unsharded --output_dir OLMo-150M/OLMo-150M-cosine/step${step}-hf  --no_fix_eos_token_id
#     # python convert_to_hf.py --input_dir OLMo-150M/OLMo-150M-constant/step${step}-unsharded --output_dir OLMo-150M/OLMo-150M-constant/step${step}-hf  --no_fix_eos_token_id

#     python save_tokienizer.py --tokenizer_name meta-llama/Llama-2-7b-hf --target_dir OLMo-150M/OLMo-150M-cosine/step${step}-hf
#     python save_tokienizer.py --tokenizer_name meta-llama/Llama-2-7b-hf --target_dir OLMo-150M/OLMo-150M-constant/step${step}-hf
# done

steps=(128 256 384 512 640 768 896 1024)
types=(constant)

for type in "${types[@]}";do
for step in "${steps[@]}";do
    python -m verl.model_merger merge \
        --backend fsdp \
        --local_dir checkpoints/Pretrain-sharpen/OLMo-150M-${type}-gsm8k-step60000/global_step_${step}/actor\
        --target_dir checkpoints/Pretrain-sharpen/OLMo-150M-${type}-gsm8k-step60000/global_step_${step}/actor/huggingface
done
done