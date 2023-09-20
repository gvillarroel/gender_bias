import pandas as pd
from tqdm import tqdm
tqdm.pandas()
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')

df_p = pd.read_csv("data/phrases.csv", sep="\t")

df_o = pd.read_csv("data/occupation.csv")

df_cross = df_p.join(df_o, how="cross")

def scoring(row):
    seq = unmasker(row.phrase.replace("{occupation}", f"The {row.Role}"))
    return {sq["token_str"]: sq["score"] for sq in seq}

df_final1 = pd.concat([df_cross, pd.json_normalize(df_cross.progress_apply(scoring, axis=1))], axis=1)
df_final1.to_csv("data/cross_baseline.csv")
