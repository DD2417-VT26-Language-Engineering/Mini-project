from collections import Counter, defaultdict
from pathlib import Path
import pickle

INPUT_FILE = Path("data/processed/sentences.txt")
MODEL_FILE = Path("models/ngram_model.pkl")

MODEL_FILE.parent.mkdir(exist_ok=True)

unigrams = Counter()
bigrams = defaultdict(Counter)
trigrams = defaultdict(Counter)

with INPUT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        words = line.strip().split()

        if not words:
            continue

        tokens = ["<s>", "<s>"] + words + ["</s>"]

        for i in range(2, len(tokens)):
            unigrams[tokens[i]] += 1
            bigrams[tokens[i - 1]][tokens[i]] += 1
            trigrams[(tokens[i - 2], tokens[i - 1])][tokens[i]] += 1

model = {
    "unigrams": unigrams,
    "bigrams": dict(bigrams),
    "trigrams": dict(trigrams),
}

with MODEL_FILE.open("wb") as f:
    pickle.dump(model, f)

print("Saved model to", MODEL_FILE)
print("Vocabulary size:", len(unigrams))
print("Number of bigram contexts:", len(bigrams))
print("Number of trigram contexts:", len(trigrams))