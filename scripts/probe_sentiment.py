"""Is the VAE latent z a usable 'embedding' for a BERT-style task (sentiment)?

We compare a linear probe trained on three feature sets, all on the SAME probe:
  1. z        -- the VAE text latent (encode_text -> mean, flattened)
  2. surface  -- hashed char n-grams (pure surface form, NO semantics)
  3. semantic -- a real sentence embedding (MiniLM), if sentence-transformers is
                 installed (needs a one-time download). The semantic upper bound.

Reading the result:
  - If probe(z) ~= probe(surface)  -> z carries the text at the SURFACE level; it
    behaves like raw characters, not like BERT.
  - If probe(semantic) >> both     -> that gap is the meaning z is NOT giving you.

Runs on CPU only -- it does not touch the GPU. Point --data at a CSV of
"text,label" (label in {0,1}) for real numbers; otherwise a small built-in set
is used just to illustrate the method.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from wordfusion.config import DEFAULT as cfg
from wordfusion.textio import VOCAB, text_to_tokens
from wordfusion.textimage_vae import TextImageVAE

# --- a tiny built-in sentiment set (illustrative; use --data for real numbers) ---
_POS = [
    "I absolutely loved this movie", "what a fantastic performance", "the food was delicious",
    "a wonderful and heartwarming story", "best purchase I have ever made", "this made me so happy",
    "brilliant writing and great acting", "I highly recommend it to everyone", "an amazing experience",
    "the service was excellent", "beautifully shot and deeply moving", "such a delightful surprise",
    "it exceeded all my expectations", "a masterpiece from start to finish", "pure joy to watch",
    "the staff were friendly and helpful", "gorgeous visuals and a lovely score", "I can't stop smiling",
    "genuinely one of the best I've seen", "everything about it was perfect", "a charming little gem",
    "incredibly satisfying and fun", "warm, funny, and clever", "I felt uplifted the whole time",
    "top notch quality all around", "this restaurant is a treasure", "the ending was so rewarding",
    "smart, stylish, and thrilling", "a truly memorable evening", "sweet and beautifully told",
    "the product works flawlessly", "worth every single penny", "the crowd was cheering with delight",
    "an inspiring and hopeful film", "fresh, bold, and wonderful", "I was thoroughly impressed",
    "the hotel room was spotless and cozy", "a joy from beginning to end", "endlessly entertaining",
    "not bad at all, actually great", "far better than I expected", "you will not be disappointed",
    "hard to find anything to dislike", "it never fails to make me happy", "an effortless pleasure",
    "the design is elegant and intuitive", "a stunning achievement", "captivating and rich",
]
_NEG = [
    "I really hated this movie", "what a terrible performance", "the food was disgusting",
    "a boring and pointless story", "worst purchase I have ever made", "this made me so angry",
    "awful writing and wooden acting", "I would never recommend it", "a miserable experience",
    "the service was appalling", "dull, flat, and forgettable", "such a bitter disappointment",
    "it failed to meet expectations", "a disaster from start to finish", "painful to sit through",
    "the staff were rude and unhelpful", "ugly visuals and an annoying score", "I can't stop frowning",
    "genuinely one of the worst I've seen", "everything about it was wrong", "a cheap and lazy mess",
    "deeply unsatisfying and tedious", "cold, humorless, and dumb", "I felt drained the whole time",
    "poor quality all around", "this restaurant is a nightmare", "the ending was so frustrating",
    "clumsy, ugly, and dull", "a truly regrettable evening", "cruel and badly told",
    "the product broke immediately", "a total waste of money", "the crowd was groaning in boredom",
    "a depressing and hopeless film", "stale, timid, and weak", "I was thoroughly unimpressed",
    "the hotel room was filthy and cramped", "a chore from beginning to end", "utterly exhausting",
    "not good at all, actually terrible", "far worse than I feared", "you will surely be disappointed",
    "hard to find anything to like", "it never fails to annoy me", "a joyless slog",
    "the design is clunky and confusing", "an embarrassing failure", "hollow and empty",
]


def load_data(path):
    if path and os.path.exists(path):
        texts, labels = [], []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    lab = int(row[-1])
                except ValueError:
                    continue                       # skip header / bad rows
                texts.append(row[0]); labels.append(lab)
        return texts, np.array(labels)
    texts = _POS + _NEG
    labels = np.array([1] * len(_POS) + [0] * len(_NEG))
    return texts, labels


# --- featurizers ---

def z_features(texts):
    """Encode each text with the trained VAE -> flattened latent mean."""
    ckpt = "checkpoints/textimage_vae.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing {ckpt} -- train the VAE first")
    model = TextImageVAE(cfg, vocab=VOCAB)
    model.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    model.eval()
    feats = []
    with torch.no_grad():
        for t in texts:
            tok = torch.from_numpy(text_to_tokens(t, cfg.text_max_len)).long()[None]  # (1, T)
            mean, _ = model.encode_text(tok)                            # (1, C, hw, hw)
            feats.append(mean.reshape(-1).numpy())
    return np.stack(feats)


def surface_features(texts, dim=2048):
    """Hashed char (2,3)-gram counts -> l2 normalized. Pure surface form."""
    X = np.zeros((len(texts), dim), np.float32)
    for i, t in enumerate(texts):
        s = t.lower()
        for n in (2, 3):
            for j in range(len(s) - n + 1):
                X[i, hash(s[j:j + n]) % dim] += 1.0
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X


def semantic_features(texts):
    """Real sentence embedding (MiniLM). Returns None if unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None
    try:
        m = SentenceTransformer("all-MiniLM-L6-v2")
        return np.asarray(m.encode(texts, show_progress_bar=False), np.float32)
    except Exception as e:
        print(f"  (semantic baseline skipped: {e})")
        return None


# --- linear probe (logistic regression, torch, CPU) ---

def probe(X, y, splits=8, epochs=300, seed0=0):
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    n, d = X.shape
    accs = []
    for s in range(splits):
        g = torch.Generator().manual_seed(seed0 + s)
        perm = torch.randperm(n, generator=g)
        n_te = max(1, n // 5)
        te, tr = perm[:n_te], perm[n_te:]
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6            # standardize on train
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        clf = torch.nn.Linear(d, int(y.max()) + 1)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2, weight_decay=1e-2)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(epochs):
            opt.zero_grad()
            lossf(clf(Xtr), y[tr]).backward()
            opt.step()
        with torch.no_grad():
            acc = (clf(Xte).argmax(1) == y[te]).float().mean().item()
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="CSV of 'text,label' (label 0/1)")
    ap.add_argument("--splits", type=int, default=8)
    args = ap.parse_args()

    texts, y = load_data(args.data)
    src = args.data if args.data else "built-in illustrative set"
    print(f"data: {src}  |  {len(texts)} examples, {int(y.sum())} pos / {len(y)-int(y.sum())} neg")
    print(f"chance accuracy ~= {max(y.mean(), 1-y.mean()):.3f}\n")

    rows = []
    print("encoding z (VAE latent) ...")
    rows.append(("z (VAE latent)",) + (lambda X: (X.shape[1],) + probe(X, y, args.splits))(z_features(texts)))
    print("building surface n-grams ...")
    rows.append(("surface char n-gram",) + (lambda X: (X.shape[1],) + probe(X, y, args.splits))(surface_features(texts)))
    print("trying semantic (MiniLM) ...")
    sem = semantic_features(texts)
    if sem is not None:
        rows.append(("semantic MiniLM",) + (sem.shape[1],) + probe(sem, y, args.splits))

    print("\n  features               dim    test-acc (mean +/- std)")
    print("  " + "-" * 55)
    for name, dim, mu, sd in rows:
        print(f"  {name:<22}{dim:>5}    {mu:.3f} +/- {sd:.3f}")
    print("\nread: probe(z) ~ probe(surface)  => z is surface-level (acts like raw text, not BERT).")
    print("      probe(semantic) >> both     => that gap is the meaning z does not encode.")
    if sem is None:
        print("      (install sentence-transformers to add the semantic upper bound.)")


if __name__ == "__main__":
    main()
