"""Лабораторная 1. Биграммы и BPE на тексте «Война и мир»."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
from urllib.request import urlopen


FOLDER = Path(__file__).resolve().parent
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
SEED = 42
UNKNOWN = "<UNK>"
END_OF_WORD = "</w>"


def load_words():
    path = FOLDER / "corpus.txt"
    if not path.exists():
        with urlopen(CORPUS_URL, timeout=60) as response:
            path.write_bytes(response.read())

    text = path.read_text(encoding="utf-8")
    text = text.lower().replace("ё", "е")
    return re.findall(r"[а-я]+(?:-[а-я]+)*", text)


class BigramModel:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def fit(self, words):
        self.word_counts = Counter(words)
        self.vocabulary = sorted(self.word_counts) + [UNKNOWN]
        self.context_counts = Counter()
        self.pair_counts = Counter()

        # Последнее слово не считаем контекстом: за ним ничего нет.
        for previous, current in zip(words, words[1:]):
            self.context_counts[previous] += 1
            self.pair_counts[previous, current] += 1

    def known_word(self, word):
        if word in self.word_counts:
            return word
        return UNKNOWN

    def prob(self, previous, current):
        previous = self.known_word(previous)
        current = self.known_word(current)
        pair_count = self.pair_counts[previous, current]
        context_count = self.context_counts[previous]
        # Сглаживание Лапласа даёт ненулевую вероятность новым парам.
        return (pair_count + self.alpha) / (
            context_count + self.alpha * len(self.vocabulary)
        )

    def logprob(self, words):
        total = 0.0
        for previous, current in zip(words, words[1:]):
            total += math.log(self.prob(previous, current))
        return total

    def generate(self, start, temperature, length=30):
        rng = random.Random(SEED)
        words = [start]

        for _ in range(length - 1):
            previous = words[-1]
            weights = []
            for word in self.vocabulary:
                probability = self.prob(previous, word)
                weights.append(probability ** (1 / temperature))
            # choices принимает веса и сам учитывает их сумму при выборе.
            next_word = rng.choices(self.vocabulary, weights=weights)[0]
            words.append(next_word)

        return " ".join(words)


def calculate_bigram(words, word_counts):
    split = len(words) * 9 // 10
    train = words[:split]
    validation = words[split:]
    model = BigramModel()
    model.fit(train)

    perplexity = {}
    for alpha in (1, 0.1, 0.01):
        model.alpha = alpha
        average_logprob = model.logprob(validation) / (len(validation) - 1)
        perplexity[str(alpha)] = math.exp(-average_logprob)

    original = validation[:12]
    shuffled = original.copy()
    random.Random(SEED).shuffle(shuffled)
    unseen_pairs = 0
    for previous, current in zip(original, original[1:]):
        if model.pair_counts[previous, current] == 0:
            unseen_pairs += 1

    generation = []
    for temperature in (0.5, 1.0, 2.0):
        for start in ("князь", "она"):
            generation.append({
                "temperature": temperature,
                "start": start,
                "text": model.generate(start, temperature),
            })

    return {
        "corpus": CORPUS_URL,
        "words": len(words),
        "train_words": len(train),
        "val_words": len(validation),
        "vocab_size": len(model.word_counts),
        "top20": word_counts.most_common(20),
        "perplexity": perplexity,
        "likelihood": {
            "sentence": " ".join(original),
            "shuffled": " ".join(shuffled),
            "alpha": model.alpha,
            "logprob_sentence": model.logprob(original),
            "logprob_shuffled": model.logprob(shuffled),
            "unseen_pairs_in_sentence": unseen_pairs,
        },
        "generation": generation,
    }


def merge_pair(symbols, pair):
    result = []
    i = 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            result.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


def learn_merges(word_counts, number_of_merges):
    vocabulary = []
    for word, count in sorted(word_counts.items()):
        if count >= 2:
            vocabulary.append((list(word) + [END_OF_WORD], count))

    merges = []
    for step in range(number_of_merges):
        pair_counts = Counter()
        for symbols, word_count in vocabulary:
            for left, right in zip(symbols, symbols[1:]):
                pair_counts[left, right] += word_count

        # При равенстве частот выбираем первую пару в порядке обхода.
        best_pair, _ = pair_counts.most_common(1)[0]
        merges.append(best_pair)

        for i, (symbols, word_count) in enumerate(vocabulary):
            vocabulary[i] = (merge_pair(symbols, best_pair), word_count)

        if (step + 1) % 200 == 0:
            print(f"BPE {step + 1}/{number_of_merges}", flush=True)
    return merges


def tokenize(word, ranks):
    symbols = list(word) + [END_OF_WORD]
    while len(symbols) > 1:
        best_rank = len(ranks)
        best_position = None

        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i + 1])
            rank = ranks.get(pair, len(ranks))
            if rank < best_rank:
                best_rank = rank
                best_position = i

        if best_position is None:
            break

        i = best_position
        symbols[i:i + 2] = [symbols[i] + symbols[i + 1]]
    return symbols


def calculate_bpe(word_counts, number_of_merges):
    merges = learn_merges(word_counts, number_of_merges)
    ranks = {}
    alphabet = {END_OF_WORD}
    for rank, (left, right) in enumerate(merges):
        ranks[left, right] = rank
        alphabet.add(left + right)

    word_total = 0
    char_total = 0
    token_total = 0
    for word, count in word_counts.items():
        alphabet.update(word)
        word_total += count
        char_total += len(word) * count
        token_total += len(tokenize(word, ranks)) * count

    examples = {}
    for word in ("наполеон", "сказала", "неудовольствие"):
        examples[word] = tokenize(word, ranks)

    return {
        "num_merges": len(merges),
        "vocab_size": len(alphabet),
        "merges": merges,
        "stats": {
            "chars": char_total,
            "words": word_total,
            "tokens_bpe": token_total,
            "tokens_per_word": token_total / word_total,
            "compression_vs_chars": char_total / token_total,
        },
        "examples": examples,
    }


def save_json(filename, data):
    folder = FOLDER / "artifacts"
    folder.mkdir(exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    (folder / filename).write_text(text + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    number_of_merges = 1500
    if args.fast:
        number_of_merges = 200

    words = load_words()
    word_counts = Counter(words)
    bigram = calculate_bigram(words, word_counts)
    save_json("bigram.json", bigram)
    bpe = calculate_bpe(word_counts, number_of_merges)
    save_json("bpe.json", bpe)
    print("Перплексия", bigram["perplexity"])
    print("Токенов на слово", round(bpe["stats"]["tokens_per_word"], 3))


if __name__ == "__main__":
    main()
