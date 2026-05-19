steps=(128 256 384 512 640 768 896)
steps=(1024)
types=(constant)

for type in "${types[@]}"; do
for step in "${steps[@]}"; do
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark GSM8K --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark Math-500 --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark Olympiad-Bench --train_type mix --rollout_n 16
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark AMC23 --train_type mix --rollout_n 16

done
done