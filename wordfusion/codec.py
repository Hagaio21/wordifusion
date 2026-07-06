"""Text <-> colored-image codec.

Every character in our vocabulary is assigned a distinct RGB color drawn from a
coarse 5x5x5 grid of the RGB cube. The wide spacing between grid points (64 per
channel) is deliberate: after a VAE and a diffusion model have smeared the colors
around, a nearest-neighbor lookup can still snap each pixel back to the right
character.

A string is laid out row-major into an (H, W) grid of pixels, one char per pixel,
padded/truncated to exactly H*W characters.
"""

from __future__ import annotations

import numpy as np

# --- Vocabulary ------------------------------------------------------------

# Printable ASCII (space through '~') plus newline, plus an explicit PAD token
# used to fill unused pixels.
PAD = "\x00"  # sentinel; never appears in real input text
_PRINTABLE = [chr(c) for c in range(32, 127)]  # 95 chars, ' ' .. '~'
VOCAB = [PAD, "\n"] + _PRINTABLE               # index 0 = PAD, 1 = newline, ...

CHAR_TO_IDX = {ch: i for i, ch in enumerate(VOCAB)}
NUM_TOKENS = len(VOCAB)  # 97


# --- Palette ---------------------------------------------------------------

def _build_palette(num_tokens: int) -> np.ndarray:
    """Return an (num_tokens, 3) uint8 palette of well-separated RGB colors."""
    levels = np.array([0, 64, 128, 191, 255], dtype=np.uint8)  # 5 levels -> 125 colors
    grid = np.array([(r, g, b) for r in levels for g in levels for b in levels],
                    dtype=np.uint8)
    if num_tokens > len(grid):
        raise ValueError(f"palette can hold {len(grid)} colors, need {num_tokens}")
    return grid[:num_tokens].copy()


PALETTE = _build_palette(NUM_TOKENS)  # (97, 3) uint8


# --- Encode / decode -------------------------------------------------------

def text_to_idx(text: str, height: int = 16, width: int = 16) -> np.ndarray:
    """Encode `text` into an (H, W) int64 grid of vocabulary indices.

    Characters outside the vocabulary are dropped. Padded with PAD (or truncated)
    to exactly height*width characters.
    """
    capacity = height * width
    idxs = [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]
    idxs = idxs[:capacity]
    idxs += [CHAR_TO_IDX[PAD]] * (capacity - len(idxs))
    return np.array(idxs, dtype=np.int64).reshape(height, width)


def idx_to_image(idx: np.ndarray) -> np.ndarray:
    """(H, W) index grid -> (H, W, 3) uint8 RGB image via the palette."""
    return PALETTE[idx]


def idx_to_text(idx: np.ndarray, strip_pad: bool = True) -> str:
    """(H, W) index grid -> text (row-major)."""
    flat = np.asarray(idx).reshape(-1)
    text = "".join(VOCAB[i] for i in flat)
    return text.replace(PAD, "") if strip_pad else text


def encode_text(text: str, height: int = 16, width: int = 16) -> np.ndarray:
    """Encode `text` into an (H, W, 3) uint8 image."""
    return idx_to_image(text_to_idx(text, height, width))


def image_to_idx(image: np.ndarray) -> np.ndarray:
    """Decode an (H, W, 3) image to an (H, W) index grid via nearest-palette lookup.

    Works on noisy images (floats or uint8): each pixel maps to the closest
    palette color in Euclidean RGB distance.
    """
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    flat = img.reshape(-1, 3)
    dists = ((flat[:, None, :] - PALETTE[None, :, :].astype(np.float32)) ** 2).sum(-1)
    return dists.argmin(axis=1).reshape(h, w)


def decode_image(image: np.ndarray, strip_pad: bool = True) -> str:
    """Decode an (H, W, 3) image back to text via nearest-palette lookup."""
    return idx_to_text(image_to_idx(image), strip_pad=strip_pad)


def upscale(image: np.ndarray, factor: int = 16) -> np.ndarray:
    """Nearest-neighbor upscale so 1 char becomes a visible block. For viewing."""
    return np.repeat(np.repeat(image, factor, axis=0), factor, axis=1)
