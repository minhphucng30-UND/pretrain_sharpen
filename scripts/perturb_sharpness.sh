export CUDA_VISIBLE_DEVICES=3

steps=(3000 6000 9000 12000 15000 18000 21000 24000 27000 30000 33000 36000 39000 42000 45000 48000 51000 54000 57000 60000 61777)
epsilons=(1e-3 2.5e-3 5e-3 1e-2)
# datasets=(tinygsm openmathinstruct-1 openmathinstruct-2 proofpile finemath-3plus)
datasets=(finemath-3plus)
scheduler_types=(cosine constant)

for epsilon in ${epsilons[@]}; do
for dataset in ${datasets[@]}; do
for scheduler_type in ${scheduler_types[@]}; do
for step in ${steps[@]}; do
    python analysis/get_sharpness.py --step $step --epsilon $epsilon --dataset_name $dataset --scheduler_type ${scheduler_type}
done
done
done
done