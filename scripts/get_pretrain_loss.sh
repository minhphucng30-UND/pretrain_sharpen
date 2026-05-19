export CUDA_VISIBLE_DEVICES=0

steps=(128 256 384 512 640 768 896 1024)
datasets=(tinygsm openmathinstruct-1 openmathinstruct-2 proofpile finemath-3plus)
scheduler_types=(constant constant-6e4 constant-1e3 constant-3e3 cosine)

for dataset in ${datasets[@]}; do
for scheduler_type in ${scheduler_types[@]}; do
for step in ${steps[@]}; do
    python analysis/get_pretrain_loss.py --step $step --dataset_name $dataset --scheduler_type ${scheduler_type}

done
done
done