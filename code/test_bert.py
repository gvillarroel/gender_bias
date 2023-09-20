from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased').cuda()
OCUPATION = "Conveyance Operator"

def scores(occupation):
    seq = unmasker(f"{occupation} went out because [MASK] needs some fresh air")
    return {sq["token_str"]: sq["score"] for sq in seq}

import pandas as pd

df = pd.read_csv("occupation.csv")


df2 = pd.concat([df, pd.json_normalize(df["R"].apply(scores))], axis=1)

df2.to_csv("data.csv")