import re
from pathlib import Path

INPUT_FILE = Path("data/processed/dialogues.txt")
OUTPUT_FILE = Path("data/processed/sentences.txt")

text = INPUT_FILE.read_text(encoding="utf-8").lower()

# Normalize unicode apostrophes
text = text.replace("’", "'")

# Fix spaced contractions
text = re.sub(r"\b([a-z]+)\s+'\s+([a-z]+)\b", r"\1'\2", text)

text = text.replace("[", " ").replace("]", " ")
text = text.replace('"', " ")

chunks = re.split(r"[.!?]+", text)

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    for chunk in chunks:
        chunk = re.sub(r"[^a-z'\-\s]", " ", chunk)

        # Remove apostrophes not inside words
        chunk = re.sub(r"(?<![a-z])'|'(?![a-z])", " ", chunk)

        # Remove hyphens not inside words
        chunk = re.sub(r"(?<![a-z])-|-(?![a-z])", " ", chunk)

        # Normalize whitespace
        chunk = re.sub(r"\s+", " ", chunk).strip()

        if len(chunk.split()) >= 2:
            out.write(chunk + "\n")

print(f"Saved cleaned sentences to {OUTPUT_FILE}")