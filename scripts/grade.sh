steps=(1024)
types=(constant-6e4)

for type in "${types[@]}"; do
for step in "${steps[@]}"; do
    # python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark GSM8K
    python evals/grade.py --step ${step} --model_name OLMo-150M --type ${type} --benchmark Math-500
done
done