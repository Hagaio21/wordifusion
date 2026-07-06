"""Character-level ASCII text<->token I/O + training data (no tokenizer).

Fixed alphabet: newline + printable ASCII (' '..'~') = 96 characters, plus a PAD
token = 97 total. One token per character. Anything outside the alphabet (other
Unicode, control chars) is mapped to '?'. No learned tokenization -- that's a
later problem.

To avoid overfitting to one corpus, training can mix real text (every .txt in the
data dir) with random-character windows (`random_frac`).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

# --- fixed ASCII alphabet ---
ALPHABET = "\n" + "".join(chr(c) for c in range(32, 127))   # 96 chars: '\n' + ' '..'~'
CHAR_TO_ID = {ch: i for i, ch in enumerate(ALPHABET)}
PAD_ID = len(ALPHABET)          # 96
VOCAB = len(ALPHABET) + 1       # 97  (alphabet + PAD)
UNK = CHAR_TO_ID["?"]           # out-of-alphabet chars fold to '?'
N_CHARS = len(ALPHABET)         # number of real (non-PAD) symbols

# ASCII code -> token id lookup (default UNK); only 0..127 are addressable.
_LUT = np.full(128, UNK, dtype=np.int64)
for _ch, _i in CHAR_TO_ID.items():
    _LUT[ord(_ch)] = _i


def _encode_ascii(text: str) -> np.ndarray:
    """str -> int64 token ids (non-ASCII -> '?'), no padding."""
    b = text.encode("ascii", errors="replace")      # non-ascii -> b'?'
    return _LUT[np.frombuffer(b, dtype=np.uint8)]


def text_to_tokens(text: str, length: int) -> np.ndarray:
    """str -> (length,) int64 token ids, PAD-padded / truncated."""
    ids = _encode_ascii(text)[:length]
    if len(ids) < length:
        ids = np.concatenate([ids, np.full(length - len(ids), PAD_ID, np.int64)])
    return ids


def tokens_to_text(arr) -> str:
    """token ids -> str (drop PAD)."""
    a = np.asarray(arr).reshape(-1)
    return "".join(ALPHABET[i] for i in a if i != PAD_ID and 0 <= i < N_CHARS)


class CharWindows(Dataset):
    """Variable-length ASCII character windows, mixing real corpora + random chars.

    Each item is a length-`max_len` int64 tensor: a content run of L chars (L in
    [min_len, max_len], or fixed to max_len if var_len=False) followed by PAD. With
    probability `random_frac` the content is random alphabet characters (de-overfit
    / a max-entropy stress test); otherwise a random slice of the real corpora.
    """

    def __init__(self, data_dir: str, max_len: int, epoch_size: int = 20000,
                 random_frac: float = 0.5, min_len: int = 8, var_len: bool = True,
                 seed: int = 0):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.txt")))
        if not paths:
            raise FileNotFoundError(f"no .txt corpora found in {data_dir}/")
        text = "".join(open(p, encoding="utf-8", errors="ignore").read() for p in paths)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.data = _encode_ascii(text)             # whole corpus as token ids
        self.max_len = max_len
        self.min_len = min(min_len, max_len)
        self.var_len = var_len
        self.epoch_size = epoch_size
        self.random_frac = random_frac
        self.rng = np.random.default_rng(seed)
        self.max_start = max(1, len(self.data) - max_len)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, i: int):
        L = int(self.rng.integers(self.min_len, self.max_len + 1)) if self.var_len else self.max_len
        if self.rng.random() < self.random_frac:
            content = self.rng.integers(0, N_CHARS, size=L).astype(np.int64)
        else:
            s = int(self.rng.integers(0, self.max_start))
            content = self.data[s:s + L]
        w = np.full(self.max_len, PAD_ID, dtype=np.int64)
        w[:len(content)] = content
        return torch.from_numpy(w).long()
