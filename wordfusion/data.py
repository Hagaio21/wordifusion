"""Dataset: slide a window over the corpus and render each window as a colored image.

Each sample yields:
  rgb : float tensor (3, H, W) normalized to [-1, 1]  -- VAE / diffusion input
  idx : long  tensor (H, W)    vocabulary indices      -- cross-entropy target
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import text_to_idx, idx_to_image, CHAR_TO_IDX, PAD, PALETTE, NUM_TOKENS

# Palette in the same 0..255 space, as a tensor, for GPU rendering / decoding.
_PALETTE_T = torch.tensor(PALETTE, dtype=torch.float32)


def render_rgb(idx: torch.Tensor, device) -> torch.Tensor:
    """(B, H, W) long char indices -> (B, 3, H, W) float image in [-1, 1], on GPU.

    Rendering the whole batch on-device is far cheaper than doing it per-sample on
    the CPU, so data loading just slices index windows and this does the coloring.
    """
    pal = _PALETTE_T.to(device)                 # (NUM_TOKENS, 3) in 0..255
    rgb = pal[idx.to(device)]                    # (B, H, W, 3)
    return rgb.permute(0, 3, 1, 2).contiguous() / 127.5 - 1.0


def nearest_char_idx(rgb_tensor: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) image in [-1, 1] -> (B, H, W) nearest-palette char indices.

    This is the 'text decoder' applied to a (possibly reconstructed / noisy)
    image, done on-device so we can monitor char accuracy during training.
    """
    x = (rgb_tensor + 1.0) * 127.5                      # -> 0..255
    b, c, h, w = x.shape
    pix = x.permute(0, 2, 3, 1).reshape(-1, 3)          # (N, 3)
    pal = _PALETTE_T.to(x.device)
    d = (pix[:, None, :] - pal[None, :, :]).pow(2).sum(-1)
    return d.argmin(1).reshape(b, h, w)


def rgb_to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> (3, H, W) float in [-1, 1]."""
    t = torch.from_numpy(img_uint8.astype(np.float32) / 127.5 - 1.0)
    return t.permute(2, 0, 1).contiguous()


def tensor_to_rgb(t: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [-1, 1] -> (H, W, 3) uint8."""
    x = ((t.detach().cpu().float().permute(1, 2, 0) + 1.0) * 127.5)
    return x.clamp(0, 255).round().to(torch.uint8).numpy()


class ShakespeareWindows(Dataset):
    """Random fixed-size character windows over a text corpus, as images.

    length is virtual: we sample `epoch_size` random windows per epoch so the
    DataLoader has a well-defined length while still covering the corpus densely.
    """

    def __init__(self, path: str, img_h: int, img_w: int, epoch_size: int = 20000,
                 seed: int = 0):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        # keep only in-vocab characters so windows are dense with real text
        text = "".join(ch for ch in raw if ch in CHAR_TO_IDX and ch != PAD)
        # preconvert the whole corpus to vocab indices ONCE (vectorized), so a
        # window is just a slice -- no per-sample python loop.
        lut = np.full(256, CHAR_TO_IDX[PAD], dtype=np.int64)
        for ch, i in CHAR_TO_IDX.items():
            if len(ch) == 1 and ord(ch) < 256:
                lut[ord(ch)] = i
        self.codes = lut[np.frombuffer(text.encode("latin-1"), dtype=np.uint8)]
        self.h, self.w = img_h, img_w
        self.win = img_h * img_w
        self.epoch_size = epoch_size
        self.rng = np.random.default_rng(seed)
        self.max_start = max(1, len(self.codes) - self.win)

    def __len__(self) -> int:
        return self.epoch_size

    def __getitem__(self, i: int):
        # random window; returns only the index grid (rgb is rendered on GPU).
        start = int(self.rng.integers(0, self.max_start))
        window = self.codes[start:start + self.win].reshape(self.h, self.w)
        return torch.from_numpy(window.copy()).long()
