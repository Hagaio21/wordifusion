"""Learn a SEMANTIC color per character via skip-gram (char2vec) in 3-D.

Each character's embedding is trained to predict its neighboring characters, so
characters used in similar contexts get similar embeddings. With a 3-D embedding,
those three numbers are the RGB color -> a learned, semantic palette.

Outputs:
  - checkpoints/char_colors.npy   (97, 3) uint8 learned palette
  - samples/learned_palette.png   labeled grid, ordered so similar chars are adjacent
  - prints nearest-neighbor characters for a few queries
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from wordfusion.config import DEFAULT as cfg
from wordfusion.codec import CHAR_TO_IDX, PAD, VOCAB, NUM_TOKENS

DIM = 3          # 3-D embedding == RGB color
WINDOW = 5
STEPS = 8000
BATCH = 8192
device = "cpu"   # tiny model; keep off the GPU so it won't disturb training there

# --- corpus -> indices ---
raw = open(cfg.corpus_path, encoding="utf-8").read()
lut = np.full(256, CHAR_TO_IDX[PAD], dtype=np.int64)
for ch, i in CHAR_TO_IDX.items():
    if len(ch) == 1 and ord(ch) < 256:
        lut[ord(ch)] = i
codes = lut[np.frombuffer("".join(c for c in raw if c in CHAR_TO_IDX and c != PAD)
                          .encode("latin-1"), dtype=np.uint8)]
N = len(codes)
print(f"corpus {N} chars, vocab {NUM_TOKENS}, embedding dim {DIM}")

# --- skip-gram with full 97-way softmax (vocab is tiny) ---
emb = nn.Embedding(NUM_TOKENS, DIM).to(device)
head = nn.Linear(DIM, NUM_TOKENS).to(device)
opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=5e-3)
rng = np.random.default_rng(0)

for step in range(1, STEPS + 1):
    centers = rng.integers(WINDOW, N - WINDOW, size=BATCH)
    offs = rng.integers(1, WINDOW + 1, size=BATCH) * rng.choice([-1, 1], size=BATCH)
    x = torch.from_numpy(codes[centers]).to(device)
    y = torch.from_numpy(codes[centers + offs]).to(device)
    logits = head(emb(x))
    loss = F.cross_entropy(logits, y)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 1000 == 0:
        print(f"  step {step}/{STEPS}  loss {loss.item():.3f}")

E = emb.weight.detach().cpu().numpy()                       # (97, 3)
# normalize each channel to 0..255 for a viewable color
lo, hi = E.min(0), E.max(0)
colors = ((E - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
os.makedirs("checkpoints", exist_ok=True)
np.save("checkpoints/char_colors.npy", colors)

# --- nearest-neighbor report (semantic sanity) ---
def neighbors(ch, k=6):
    i = CHAR_TO_IDX[ch]
    d = ((E - E[i]) ** 2).sum(1)
    order = np.argsort(d)[1:k + 1]
    return "".join(VOCAB[j] for j in order)

print("\nnearest chars in learned space (should look semantically grouped):")
for q in ["a", "e", "5", ".", " ", "A", "z", "!"]:
    disp = repr(q)
    print(f"  {disp:>5} -> {neighbors(q)!r}")

# --- labeled grid, ordered by 1-D PCA so similar chars sit next to each other ---
Ec = E - E.mean(0)
u, s, vt = np.linalg.svd(Ec, full_matrices=False)
order = np.argsort(Ec @ vt[0])                              # project to top PC, sort
os.makedirs("samples", exist_ok=True)


def font(sz):
    for p in (r"C:\Windows\Fonts\consolab.ttf", r"C:\Windows\Fonts\consola.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


CELL, COLS = 46, 13
rows = (NUM_TOKENS + COLS - 1) // COLS
img = Image.new("RGB", (COLS * CELL, rows * CELL), (10, 10, 10))
d = ImageDraw.Draw(img); f = font(22)
for pos, j in enumerate(order):
    r, c = divmod(pos, COLS)
    col = tuple(int(v) for v in colors[j])
    d.rectangle([c * CELL, r * CELL, c * CELL + CELL - 2, r * CELL + CELL - 2], fill=col)
    ch = VOCAB[j]
    label = {"\n": "\\n", PAD: "PAD", " ": "SP"}.get(ch, ch)
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    tc = (0, 0, 0) if lum > 140 else (255, 255, 255)
    d.text((c * CELL + 5, r * CELL + 9), label, font=f, fill=tc)
img = img.resize((COLS * CELL * 2, rows * CELL * 2), Image.NEAREST)
img.save("samples/learned_palette.png")
print("\nsaved checkpoints/char_colors.npy and samples/learned_palette.png")
print("(palette ordered by top principal component -> similar chars are adjacent)")
