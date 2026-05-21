import pandas as pd

INPUT_FILE = "data/raw/train.csv"
OUTPUT_FILE = "data/processed/dialogues.txt"

df = pd.read_csv(INPUT_FILE)

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for dialogue in df["dialog"]:
        cleaned = str(dialogue).replace("__eou__", " ")
        out.write(cleaned.lower() + "\n")

print("Done preprocessing.")