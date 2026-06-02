# Word Prediction Mini-project

This project implements a simple word prediction system using two different language modelling approaches:

- an interpolated n-gram model based on unigram, bigram, and trigram counts
- a small Transformer language model using subword tokenization

The models suggest whole-word completions while the user types. A small web GUI is included so the predictors can be tested interactively.

## Project Structure

```text
.
├── data/
│   ├── sentences.txt
│   ├── token_ids.json
│   └── processed/
│       ├── dialogues.txt
│       └── sentences.txt
├── models/
│   └── ngram_model.pkl
├── src/
│   ├── preprocess.py
│   ├── preprocess_clean.py
│   ├── train_ngram.py
│   └── predict.py
├── self_attention.py
├── transformer_model.py
├── word_gui.py
├── tokenizer.json
└── word_predictor_15_epochs.pth
```

## Setup

Install the dependencies:

```bash
pip install -r requirements.txt
pip install torch tokenizers
```

## N-gram Model

The n-gram model is trained from `data/processed/sentences.txt`. It stores unigram, bigram, and trigram counts and ranks candidate words using interpolation:

```text
0.7 * trigram_probability
+ 0.2 * bigram_probability
+ 0.1 * unigram_probability
```

To retrain the n-gram model:

```bash
python3 src/train_ngram.py
```

To try the n-gram predictor in the terminal:

```bash
python3 src/predict.py
```

## Transformer Model

The Transformer model is defined in `transformer_model.py` and uses the self-attention implementation in `self_attention.py`. It predicts subword tokens, which are then filtered into valid whole-word suggestions for the word prediction task.

The GUI loads `word_predictor_15_epochs.pth` by default. A different checkpoint can be passed with:

```bash
python3 word_gui.py --checkpoint path/to/checkpoint.pth
```

## Running the GUI

Start the interactive word prediction interface with:

```bash
python3 word_gui.py
```

Then open:

```text
http://127.0.0.1:8000
```

The GUI contains a text editor and a toggle for switching between the n-gram model and the Transformer model. Suggestions update after each keystroke, and the current inline suggestion can be accepted with the Tab key.
