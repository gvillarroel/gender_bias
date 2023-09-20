import pandas as pd
df = pd.read_csv("../data/cross_baseline.csv")
df_clean1 = df.dropna(subset=["she_prob","he_prob"])
