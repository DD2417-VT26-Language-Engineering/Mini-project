import argparse
import errno
import json
import pickle
import re
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import torch

from transformer_model import (
    Config,
    TinyStoriesLM,
    get_tokenizer,
    is_valid_suggestion_word,
    load_word_vocabulary,
)


NGRAM_MODEL_FILE = Path("models/ngram_model.pkl")
NGRAM_TEXT_FILE = Path("data/processed/sentences.txt")
DEFAULT_TRANSFORMER_CHECKPOINT = Path("word_predictor_6_epochs.pth")
WORD_RE = re.compile(r"[a-z]+(?:['-][a-z]+)*")
PREFIX_RE = re.compile(r"[a-z][a-z'-]*$")


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Word Predictor</title>
  <style>
    :root {
      color-scheme: light;
      --background: #eadfcd;
      --surface: #fffaf1;
      --surface-border: #d8c7ad;
      --text: #2a2520;
      --muted: #a9a9a9;
      --accent: #345b52;
      --accent-soft: #dce8e1;
      --shadow: 0 24px 70px rgba(63, 45, 25, 0.16);
    }

    * {
      box-sizing: border-box;
    }

    html,
    body {
      height: 100%;
      margin: 0;
    }

    body {
      display: grid;
      place-items: center;
      background: var(--background);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    .app {
      width: min(900px, calc(100vw - 32px));
    }

    .toolbar {
      display: flex;
      justify-content: flex-end;
      margin-bottom: 14px;
    }

    .model-toggle {
      display: inline-grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px;
      padding: 4px;
      border: 1px solid rgba(52, 91, 82, 0.22);
      border-radius: 999px;
      background: rgba(255, 250, 241, 0.68);
      box-shadow: 0 8px 24px rgba(63, 45, 25, 0.08);
    }

    .model-toggle button {
      min-width: 112px;
      height: 36px;
      border: 0;
      border-radius: 999px;
      background: transparent;
      color: #5d544a;
      cursor: pointer;
      font: 650 14px/1 Inter, ui-sans-serif, system-ui, sans-serif;
      letter-spacing: 0;
      transition: background 160ms ease, color 160ms ease;
    }

    .model-toggle button.active {
      background: var(--accent);
      color: white;
    }

    .editor-shell {
      position: relative;
      min-height: 360px;
      border: 1px solid var(--surface-border);
      border-radius: 22px;
      background: var(--surface);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .ghost,
    textarea {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 30px 32px;
      border: 0;
      font: 500 28px/1.55 ui-serif, Georgia, Cambria, "Times New Roman", serif;
      letter-spacing: 0;
      white-space: pre-wrap;
      overflow-wrap: break-word;
    }

    .ghost {
      pointer-events: none;
      color: transparent;
      overflow: hidden;
    }

    .ghost .completion {
      color: var(--muted);
    }

    textarea {
      resize: none;
      outline: none;
      background: transparent;
      color: var(--text);
      caret-color: var(--accent);
    }

    textarea::selection {
      background: var(--accent-soft);
    }

    @media (max-width: 680px) {
      .app {
        width: min(100vw - 20px, 900px);
      }

      .toolbar {
        justify-content: center;
      }

      .model-toggle {
        width: 100%;
      }

      .model-toggle button {
        min-width: 0;
      }

      .editor-shell {
        min-height: 68vh;
        border-radius: 18px;
      }

      .ghost,
      textarea {
        padding: 22px;
        font-size: 22px;
      }
    }
  </style>
</head>
<body>
  <main class="app">
    <div class="toolbar">
      <div class="model-toggle" role="group" aria-label="Model">
        <button type="button" class="active" data-model="ngram">n-gram</button>
        <button type="button" data-model="transformer">transformer</button>
      </div>
    </div>

    <div class="editor-shell">
      <div class="ghost" aria-hidden="true">
        <span id="ghostMirror"></span><span id="ghostCompletion" class="completion"></span>
      </div>
      <textarea id="editor" aria-label="Text editor" autocomplete="off" autocorrect="off" autocapitalize="sentences" spellcheck="false" autofocus></textarea>
    </div>
  </main>

  <script>
    const editor = document.querySelector("#editor");
    const mirror = document.querySelector("#ghostMirror");
    const completion = document.querySelector("#ghostCompletion");
    const buttons = [...document.querySelectorAll("[data-model]")];

    let activeModel = "ngram";
    let currentCompletion = "";
    let debounceTimer = null;
    let requestId = 0;
    let controller = null;

    function renderGhost() {
      mirror.textContent = editor.value;
      completion.textContent = currentCompletion;
    }

    function setModel(model) {
      activeModel = model;
      buttons.forEach((button) => {
        button.classList.toggle("active", button.dataset.model === model);
      });
      requestSuggestion();
    }

    function requestSuggestion() {
      window.clearTimeout(debounceTimer);
      debounceTimer = window.setTimeout(fetchSuggestion, 45);
    }

    async function fetchSuggestion() {
      const id = ++requestId;
      if (controller) {
        controller.abort();
      }

      controller = new AbortController();
      const params = new URLSearchParams({
        model: activeModel,
        text: editor.value,
        k: "5",
      });

      try {
        const response = await fetch(`/api/suggest?${params.toString()}`, {
          signal: controller.signal,
        });
        const data = await response.json();

        if (id !== requestId) {
          return;
        }

        currentCompletion = data.completion || "";
        renderGhost();
      } catch (error) {
        if (error.name !== "AbortError") {
          currentCompletion = "";
          renderGhost();
        }
      }
    }

    editor.addEventListener("input", () => {
      currentCompletion = "";
      renderGhost();
      requestSuggestion();
    });

    editor.addEventListener("scroll", () => {
      document.querySelector(".ghost").scrollTop = editor.scrollTop;
      document.querySelector(".ghost").scrollLeft = editor.scrollLeft;
    });

    editor.addEventListener("keydown", (event) => {
      if (
        event.key === "Tab" &&
        currentCompletion &&
        editor.selectionStart === editor.value.length &&
        editor.selectionEnd === editor.value.length
      ) {
        event.preventDefault();
        editor.value += currentCompletion;
        currentCompletion = "";
        renderGhost();
        requestSuggestion();
      }
    });

    buttons.forEach((button) => {
      button.addEventListener("click", () => setModel(button.dataset.model));
    });

    renderGhost();
    requestSuggestion();
  </script>
</body>
</html>
"""


def normalize_text(text):
    return (
        text.lower()
        .replace("’", "'")
        .replace("‘", "'")
        .replace("`", "'")
    )


def split_context_and_prefix(text):
    text = normalize_text(text)
    prefix_match = PREFIX_RE.search(text)

    if prefix_match:
        prefix = prefix_match.group(0)
        context_text = text[:prefix_match.start()]
    else:
        prefix = ""
        context_text = text

    return WORD_RE.findall(context_text), prefix


def completion_for(text, suggestion):
    if not suggestion:
        return ""

    _, prefix = split_context_and_prefix(text)
    if prefix:
        return suggestion[len(prefix):] if suggestion.startswith(prefix) else ""

    return suggestion


def prob_from_counter(counter, word):
    total = sum(counter.values())
    if total == 0:
        return 0

    return counter[word] / total


class NgramPredictor:
    def __init__(self, unigrams, bigrams, trigrams):
        self.unigrams = unigrams
        self.bigrams = bigrams
        self.trigrams = trigrams
        self.total_unigrams = sum(unigrams.values())
        self.vocabulary = [
            word
            for word, _ in unigrams.most_common()
            if word != "</s>" and is_valid_suggestion_word(word)
        ]

    @classmethod
    def load(cls, model_file=NGRAM_MODEL_FILE, text_file=NGRAM_TEXT_FILE):
        if model_file.exists():
            with model_file.open("rb") as f:
                model = pickle.load(f)
            return cls(model["unigrams"], model["bigrams"], model["trigrams"])

        unigrams = Counter()
        bigrams = defaultdict(Counter)
        trigrams = defaultdict(Counter)

        with text_file.open("r", encoding="utf-8") as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue

                tokens = ["<s>", "<s>"] + words + ["</s>"]
                for i in range(2, len(tokens)):
                    unigrams[tokens[i]] += 1
                    bigrams[tokens[i - 1]][tokens[i]] += 1
                    trigrams[(tokens[i - 2], tokens[i - 1])][tokens[i]] += 1

        return cls(unigrams, dict(bigrams), dict(trigrams))

    def predict(self, text, k=5):
        context_words, prefix = split_context_and_prefix(text)
        suggestions = self._interpolated_predict(context_words, prefix, k)

        if suggestions:
            return suggestions

        return self._interpolated_predict(context_words, "", k)

    def _interpolated_predict(self, context_words, prefix="", k=5):
        scores = {}

        for word in self.vocabulary:
            if prefix and not word.startswith(prefix):
                continue

            p_uni = self.unigrams[word] / self.total_unigrams

            p_bi = 0
            if context_words:
                prev_word = context_words[-1]
                if prev_word in self.bigrams:
                    p_bi = prob_from_counter(self.bigrams[prev_word], word)

            p_tri = 0
            if len(context_words) >= 2:
                context = (context_words[-2], context_words[-1])
                if context in self.trigrams:
                    p_tri = prob_from_counter(self.trigrams[context], word)

            scores[word] = 0.7 * p_tri + 0.2 * p_bi + 0.1 * p_uni

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [word for word, _ in ranked[:k]]


class TransformerPredictor:
    def __init__(self, checkpoint_file=DEFAULT_TRANSFORMER_CHECKPOINT, max_words_per_first_token=2):
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.tokenizer = get_tokenizer()
        self.config = Config()
        self.model = TinyStoriesLM(self.config)
        self.model.load_state_dict(torch.load(checkpoint_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.max_words_per_first_token = max_words_per_first_token
        self.word_candidates = self._encode_vocabulary(load_word_vocabulary())
        self.candidates_by_first_token = self._group_by_first_token(self.word_candidates)

    def _encode_vocabulary(self, vocabulary):
        encoded = []
        seen = set()

        for word in vocabulary:
            if word in seen:
                continue

            token_ids = self.tokenizer.encode(word).ids
            if not token_ids:
                continue

            seen.add(word)
            encoded.append((word, token_ids))

        return encoded

    def _group_by_first_token(self, word_candidates):
        groups = defaultdict(list)

        for index, (word, token_ids) in enumerate(word_candidates):
            groups[token_ids[0]].append((index, word, token_ids))

        return {
            token_id: [
                (word, token_ids)
                for _, word, token_ids in sorted(
                    candidates,
                    key=lambda item: (len(item[2]) != 1, item[0]),
                )
            ]
            for token_id, candidates in groups.items()
        }

    def predict(self, text, k=5):
        context_words, prefix = split_context_and_prefix(text)

        if not context_words:
            return [
                word
                for word, _ in self.word_candidates
                if word.startswith(prefix)
            ][:k]

        encoded_context = self.tokenizer.encode(" ".join(context_words)).ids[-self.config.block_size:]
        if not encoded_context:
            return []

        matching_first_tokens = {
            token_ids[0]
            for word, token_ids in self.word_candidates
            if word.startswith(prefix)
        }
        if not matching_first_tokens:
            return []

        with torch.no_grad():
            x = torch.tensor(encoded_context).unsqueeze(0).to(self.device)
            logits = self.model(x)
            first_log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            first_token_scores = [
                (first_log_probs[token_id].item(), token_id)
                for token_id in matching_first_tokens
            ]
            first_token_scores.sort(key=lambda item: item[0], reverse=True)

        suggestions = []
        seen = set()

        for _, first_token_id in first_token_scores:
            words_from_token = 0
            for word, _ in self.candidates_by_first_token[first_token_id]:
                if word in seen or not word.startswith(prefix):
                    continue

                seen.add(word)
                suggestions.append(word)
                words_from_token += 1

                if len(suggestions) >= k:
                    return suggestions
                if words_from_token >= self.max_words_per_first_token:
                    break

        return suggestions


class PredictorHub:
    def __init__(self, checkpoint_file):
        print("Loading n-gram model...")
        self.ngram = NgramPredictor.load()
        print("Loading transformer model...")
        self.transformer = TransformerPredictor(checkpoint_file)
        print("Models ready.")

    def suggest(self, model_name, text, k=5):
        if model_name == "transformer":
            suggestions = self.transformer.predict(text, k)
        else:
            suggestions = self.ngram.predict(text, k)

        suggestion = suggestions[0] if suggestions else ""
        return {
            "suggestions": suggestions,
            "suggestion": suggestion,
            "completion": completion_for(text, suggestion),
        }


class RequestHandler(BaseHTTPRequestHandler):
    hub = None

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
            return

        if parsed.path == "/api/suggest":
            query = parse_qs(parsed.query)
            model_name = query.get("model", ["ngram"])[0]
            text = query.get("text", [""])[0]
            try:
                k = int(query.get("k", ["5"])[0])
            except ValueError:
                k = 5

            payload = self.hub.suggest(model_name, text, k)
            body = json.dumps(payload).encode("utf-8")

            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=str(DEFAULT_TRANSFORMER_CHECKPOINT))
    args = parser.parse_args()

    try:
        server = HTTPServer((args.host, args.port), RequestHandler)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            raise SystemExit(
                f"Port {args.port} is already in use. "
                f"Stop the other server or run: python3 word_gui.py --port {args.port + 1}"
            ) from None
        raise

    RequestHandler.hub = PredictorHub(Path(args.checkpoint))
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
