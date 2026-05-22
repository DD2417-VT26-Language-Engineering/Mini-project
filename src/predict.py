from pathlib import Path
import pickle

MODEL_FILE = Path("models/ngram_model.pkl")

with MODEL_FILE.open("rb") as f:
    model = pickle.load(f)

unigrams = model["unigrams"]
bigrams = model["bigrams"]
trigrams = model["trigrams"]

def prob_from_counter(counter, word):
    total = sum(counter.values())
    if total == 0:
        return 0
    return counter[word] / total

def interpolated_predict(context_words, prefix="", k=5):
    vocabulary = unigrams.keys()
    scores = {}

    total_unigrams = sum(unigrams.values())

    for word in vocabulary:
        if word == "</s>":
            continue

        if prefix and not word.startswith(prefix):
            continue

        # unigram probability
        p_uni = unigrams[word] / total_unigrams

        # bigram probability
        p_bi = 0
        if len(context_words) >= 1:
            prev_word = context_words[-1]
            if prev_word in bigrams:
                p_bi = prob_from_counter(bigrams[prev_word], word)

        # trigram probability
        p_tri = 0
        if len(context_words) >= 2:
            context = (context_words[-2], context_words[-1])
            if context in trigrams:
                p_tri = prob_from_counter(trigrams[context], word)

        score = 0.7 * p_tri + 0.2 * p_bi + 0.1 * p_uni
        scores[word] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [word for word, score in ranked[:k]]


def predict(text, k=5):
    text = text.lower()
    words = text.split()

    if text.endswith(" "):
        prefix = ""
        context_words = words
    else:
        prefix = words[-1] if words else ""
        context_words = words[:-1]

    suggestions = interpolated_predict(context_words, prefix, k)

    if suggestions:
        return suggestions

    return interpolated_predict(context_words, "", k)


while True:
    user_input = input("Type text: ")
    print(predict(user_input))