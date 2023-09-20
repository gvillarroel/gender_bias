import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/cross_baseline.csv")

# Get prob
df['sum_prob'] = df['she'].fillna(0) + df['he'].fillna(0)

# Calcular las nuevas columnas ajustadas
df['she_prob'] = df['she'].fillna(0) / df['sum_prob']
df['he_prob'] = df['he'].fillna(0) / df['sum_prob']

# df_clean = df.fillna({'she_prob': 0.5, 'he_prob': 0.5})
df_clean = df.dropna(subset=["she_prob","he_prob"])

import numpy as np
conditions = [
    df_clean["she"] > 0.65,
    df_clean["she"] < 0.35
]
nums = [1, -1]
df_clean["strat"] = np.select(conditions, nums, default=0)
df_roles = pd.DataFrame(df_clean.groupby(["Role"])["strat"].agg(pd.Series.mode))
df_phrase = pd.DataFrame(df_clean.groupby(["phrase"])["strat"].agg(pd.Series.mode))

for i in range(3):
    df_train, hold_roles = train_test_split(df_roles, test_size=0.15, stratify=df_roles['strat'], random_state=42+i)
    df_train, hold_phrase = train_test_split(df_phrase, test_size=0.15, stratify=df_phrase['strat'], random_state=42+i)
    df_final_excluded = df_clean.loc[df_clean["phrase"].isin(hold_phrase.index) | df_clean["Role"].isin(hold_roles.index),: ]
    df_final_train = df_clean.loc[df_clean.index.isin(df_final_excluded.index) == False,:]
    df_final_test, df_final_val = train_test_split(df_final_excluded, test_size=0.15, stratify=df_final_excluded['strat'], random_state=42+i)
    df_final_train.to_csv(f"data/split/train_{i}.csv", index=False)
    df_final_test.to_csv(f"data/split/test_{i}.csv", index=False)
    df_final_val.to_csv(f"data/split/val_{i}.csv", index=False)
