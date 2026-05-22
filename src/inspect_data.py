import pandas as pd

df = pd.read_csv("data/raw/train.csv")

print(df.head())
print(df.columns)
print(df.shape)