import pandas as pd
import os
os.makedirs('data/mix', exist_ok=True)

gsm_df = pd.read_parquet('data/gsm8k/train.parquet')
math_df = pd.read_parquet('data/math/train.parquet')


mix_df = pd.concat([gsm_df, math_df]).sample(frac=1).reset_index(drop=True)
mix_df.to_parquet('data/mix/train.parquet')