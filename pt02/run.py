"""Лабораторная 2. Обучение маленького трансформера на «Войне и мире»."""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import re
import time
from urllib.request import urlopen

import torch
from torch import nn
from torch.nn import functional as F


FOLDER = Path(__file__).resolve().parent
CORPUS_URL = (
    "https://raw.githubusercontent.com/nevmenandr/word2vec-russian-novels/"
    "master/books_before/WarAndPeace.txt"
)
END = "</w>"
PROMPT = "князь андрей"
SEED = 42
CONTEXT = 128
WIDTH = 128
HEADS = 4
LAYERS = 4
HIDDEN = 4 * WIDTH
LEARNING_RATE = 0.001


def words_from(text):
    return re.findall(r"[а-я]+(?:-[а-я]+)*", text.lower().replace("ё", "е"))


def load_words():
    path = FOLDER / "corpus.txt"
    if not path.exists():
        with urlopen(CORPUS_URL, timeout=60) as response:
            path.write_bytes(response.read())
    return words_from(path.read_text(encoding="utf-8"))


class BPE:
    def __init__(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.ranks = {}
        alphabet = set("абвгдежзийклмнопрстуфхцчшщъыьэюя-")
        alphabet.add(END)
        for rank, (left, right) in enumerate(data["merges"]):
            self.ranks[left, right] = rank
            alphabet.add(left + right)
        self.tokens = sorted(alphabet)
        self.ids = {token: i for i, token in enumerate(self.tokens)}
        self.cache = {}

    def encode_word(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = list(word) + [END]
        while len(parts) > 1:
            best_rank = len(self.ranks)
            position = None
            for i in range(len(parts) - 1):
                rank = self.ranks.get((parts[i], parts[i + 1]), len(self.ranks))
                if rank < best_rank:
                    best_rank = rank
                    position = i
            if position is None:
                break
            i = position
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        ids = [self.ids[part] for part in parts]
        self.cache[word] = ids
        return ids

    def encode_words(self, words):
        ids = []
        for word in words:
            ids.extend(self.encode_word(word))
        return ids

    def decode(self, ids):
        return "".join(self.tokens[i] for i in ids).replace(END, " ").strip()


class CausalAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(WIDTH, WIDTH)
        self.key = nn.Linear(WIDTH, WIDTH)
        self.value = nn.Linear(WIDTH, WIDTH)
        self.output = nn.Linear(WIDTH, WIDTH)
        self.head_width = WIDTH // HEADS
        mask = torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1)
        self.register_buffer("future_mask", mask)

    def split_heads(self, x):
        batch, length, _ = x.shape
        return x.reshape(batch, length, HEADS, self.head_width).transpose(1, 2)

    def forward(self, x):
        batch, length, _ = x.shape
        q = self.split_heads(self.query(x))
        k = self.split_heads(self.key(x))
        v = self.split_heads(self.value(x))

        # Верхний треугольник закрывает будущие токены.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
        scores = scores.masked_fill(self.future_mask[:length, :length], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        result = weights @ v
        result = result.transpose(1, 2).reshape(batch, length, WIDTH)
        return self.output(result)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.LayerNorm(WIDTH)
        self.attention = CausalAttention()
        self.feed_forward_norm = nn.LayerNorm(WIDTH)
        self.feed_forward = nn.Sequential(
            nn.Linear(WIDTH, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, WIDTH)
        )

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class TinyTransformer(nn.Module):
    def __init__(self, bpe):
        super().__init__()
        self.bpe = bpe
        vocab_size = len(bpe.tokens)
        self.token_embedding = nn.Embedding(vocab_size, WIDTH)
        self.position_embedding = nn.Embedding(CONTEXT, WIDTH)
        self.blocks = nn.Sequential(*(Block() for _ in range(LAYERS)))
        self.last_norm = nn.LayerNorm(WIDTH)
        self.output = nn.Linear(WIDTH, vocab_size)
        # Малые начальные логиты дают лосс около ln(размер словаря).
        nn.init.normal_(self.output.weight, std=0.02)
        nn.init.zeros_(self.output.bias)

    def forward(self, ids):
        positions = torch.arange(ids.size(1), device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        return self.output(self.last_norm(self.blocks(x)))

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_p=None):
        ids = self.bpe.encode_words(words_from(prompt))
        prompt_length = len(ids)
        device = self.output.weight.device
        rng = torch.Generator(device=device).manual_seed(SEED)

        for _ in range(max_new_tokens):
            window = torch.tensor([ids[-CONTEXT:]], device=device)
            logits = self(window)[0, -1]
            if temperature == 0:
                next_id = logits.argmax().item()
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
                if top_p is None:
                    next_id = torch.multinomial(probabilities, 1, generator=rng).item()
                else:
                    ordered, order = probabilities.sort(descending=True)
                    # Оставляем и токен, на котором накопленная сумма достигает p.
                    before = ordered.cumsum(dim=0) - ordered
                    ordered = ordered.masked_fill(before >= top_p, 0)
                    index = torch.multinomial(ordered, 1, generator=rng).item()
                    next_id = order[index].item()
            ids.append(next_id)

        return self.bpe.decode(ids), ids[prompt_length:]


def windows(stream, count, rng):
    starts = torch.randint(len(stream) - CONTEXT, (count,), generator=rng)
    positions = starts.to(stream.device)[:, None]
    positions = positions + torch.arange(CONTEXT + 1, device=stream.device)
    batch = stream[positions]
    return batch[:, :-1], batch[:, 1:]


def loss(model, x, y, reduction="mean"):
    logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction=reduction
    )


@torch.no_grad()
def fixed_loss(model, batch):
    x, y = batch
    # Считаем небольшими группами, чтобы не держать все логиты в памяти.
    total = 0.0
    for i in range(0, len(x), 4):
        total += loss(model, x[i:i + 4], y[i:i + 4], reduction="sum").item()
    return total / y.numel()


def train(model, train_ids, val_ids, steps, batch_size, interval, sample_steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    rng = torch.Generator().manual_seed(SEED)
    fixed_train = windows(train_ids, 20, torch.Generator().manual_seed(SEED + 1))
    fixed_val = windows(val_ids, 20, torch.Generator().manual_seed(SEED + 2))
    curve = []
    samples = []
    accumulated = 0.0
    start = time.perf_counter()

    for step in range(steps + 1):
        if step > 0:
            model.train()
            x, y = windows(train_ids, batch_size, rng)
            optimizer.zero_grad()
            train_loss = loss(model, x, y)
            train_loss.backward()
            optimizer.step()
            accumulated += train_loss.item()

        model.eval()
        if step % interval == 0:
            if step == 0:
                train_mean = fixed_loss(model, fixed_train)
            else:
                train_mean = accumulated / interval
            val_mean = fixed_loss(model, fixed_val)
            curve.append((step, train_mean, val_mean))
            accumulated = 0.0
            print(f"Шаг {step}/{steps} | train {train_mean:.3f} | val {val_mean:.3f}", flush=True)

        if step in sample_steps:
            text, _ = model.generate(PROMPT, 60, temperature=0.8)
            samples.append({"step": step, "text": text})

    return curve, samples, (time.perf_counter() - start) / 60


def first_repeat(ids):
    seen = set()
    for i in range(len(ids) - 3):
        four = tuple(ids[i:i + 4])
        if four in seen:
            return i
        seen.add(four)
    return None


def generation_results(model, count):
    runs = []
    modes = [("greedy", 0.0, None), ("t0.7", 0.7, None),
             ("t1.5", 1.5, None), ("top_p0.9", 1.0, 0.9)]
    for name, temperature, top_p in modes:
        text, ids = model.generate(PROMPT, count, temperature, top_p)
        run = {"name": name, "temperature": temperature, "top_p": top_p,
               "text": text, "ids": ids, "n_tokens": len(ids)}
        if name == "greedy":
            run["loop_start"] = first_repeat(ids)
        runs.append(run)
    return {"model": "tiny", "prompt": PROMPT, "max_new_tokens": count, "runs": runs}


def bigram_nats(train_words, val_words, alpha):
    # Как в ЛР1, последнее слово обучения не входит в число контекстов.
    vocabulary = set(train_words)
    size = len(vocabulary) + 1
    contexts = Counter(train_words[:-1])
    pairs = Counter(zip(train_words, train_words[1:]))
    words = [word if word in vocabulary else "<UNK>" for word in val_words]
    total = 0.0
    for previous, current in zip(words, words[1:]):
        probability = (pairs[previous, current] + alpha) / (contexts[previous] + alpha * size)
        total -= math.log(probability)
    return total


@torch.no_grad()
def transformer_nats(model, words):
    ids = model.bpe.encode_words(words)
    # Первое слово служит контекстом для обеих сравниваемых моделей.
    first = len(model.bpe.encode_word(words[0]))
    device = model.output.weight.device
    total = 0.0
    for start in range(first, len(ids), CONTEXT // 2):
        end = min(start + CONTEXT // 2, len(ids))
        left = max(0, end - 1 - CONTEXT)
        x = torch.tensor([ids[left:end - 1]], device=device)
        logits = model(x)[0, start - 1 - left:]
        y = torch.tensor(ids[start:end], device=device)
        total += F.cross_entropy(logits, y, reduction="sum").item()
    return total, len(ids) - first


def evaluation(model, train_words, val_words):
    def metrics(total, tokens):
        nats = total / len(val_words)
        return {"tokens": tokens, "nats_total": total, "nats_per_word": nats,
                "ppl_per_word": math.exp(nats)}

    alpha = 0.01
    bigram = metrics(bigram_nats(train_words, val_words, alpha), len(val_words) - 1)
    bigram["alpha"] = alpha
    total, token_count = transformer_nats(model, val_words)
    return {"words": len(val_words), "bigram": bigram, "tiny": metrics(total, token_count)}


def write_json(folder, name, data):
    (folder / name).write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="200 шагов вместо 2000")
    parser.add_argument("--bpe", type=Path, default=FOLDER.parent / "pt01/artifacts/bpe.json")
    args = parser.parse_args()
    torch.manual_seed(SEED)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    words = load_words()
    split = len(words) * 9 // 10
    train_words, val_words = words[:split], words[split:]
    bpe = BPE(FOLDER / args.bpe)
    train_ids = torch.tensor(bpe.encode_words(train_words), device=device)
    val_ids = torch.tensor(bpe.encode_words(val_words), device=device)

    if args.fast:
        steps, batch, interval = 200, 16, 25
        sample_steps = (0, 50, 100, steps)
        new_tokens, eval_words = 40, 300
    else:
        steps, batch, interval = 2000, 32, 100
        sample_steps = (0, 100, 500, steps)
        new_tokens, eval_words = 80, 2000

    model = TinyTransformer(bpe).to(device)
    curve, samples, minutes = train(
        model, train_ids, val_ids, steps, batch, interval, sample_steps
    )
    generation = generation_results(model, new_tokens)
    scores = evaluation(model, train_words, val_words[:eval_words])
    config = {
        "vocab_size": len(bpe.tokens), "d_model": WIDTH, "n_layers": LAYERS,
        "n_heads": HEADS, "d_ff": HIDDEN, "context": CONTEXT,
        "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "steps": steps, "batch": batch, "lr": LEARNING_RATE, "seed": SEED,
        "train_tokens": len(train_ids), "val_tokens": len(val_ids),
        "bpe_merges": len(bpe.ranks), "train_minutes": minutes, "device": str(device),
    }

    folder = FOLDER / "artifacts"
    folder.mkdir(exist_ok=True)
    write_json(folder, "tiny_config.json", config)
    write_json(folder, "samples.json", {"prompt": PROMPT, "temperature": 0.8, "samples": samples})
    write_json(folder, "generation.json", generation)
    write_json(folder, "eval.json", scores)
    with (folder / "train_curve.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("step", "train_loss", "val_loss"))
        writer.writerows(curve)
    print(f"Готово. Результаты в {folder}", flush=True)


if __name__ == "__main__":
    main()
