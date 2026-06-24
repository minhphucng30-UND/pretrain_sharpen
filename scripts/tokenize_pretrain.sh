for src in dclm flan pes2o wiki stackexchange math; do
    python pretraining/domino_mix_to_tokens.py --step tokenize --source $src --destination data/dolmino-mix-$src-tokenized
done

python pretraining/domino_mix_to_tokens.py --step merge --destination data/dolmino-mix-50B-tokenized


for src in cranemath megamatt tinyMATH_mind tinyMATH_pot; do
  python pretraining/dolma3_mix_to_tokens.py \
    --step tokenize --source $src --destination data/dolma3-mix-$src-tokenized
done

python pretraining/dolma3_mix_to_tokens.py \
  --step merge --destination data/dolma3-mix-math-10B-tokenized