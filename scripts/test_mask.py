"""Mask part of the code-image, then try to recover the text from it.

Pipeline per trial:  text -> z -> image -> [MASK a region] -> encode -> decode -> text
We measure how well the text survives, and -- for spatial masks -- WHERE it breaks
(a per-character accuracy heatmap over the 16x16 text grid).

What it tells you:
  - If masking the LEFT of the image mainly kills the LEFT text positions, the
    encoding is LOCALIZED (pixel region <-> text region).
  - If masking anywhere degrades ALL positions roughly equally, the encoding is
    DISTRIBUTED / superposed across the whole image.
This also measures erasure-robustness (relevant to the watermark/stego angle).

CPU only; does not touch the GPU.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from wordfusion.config import DEFAULT as cfg
from wordfusion.textio import CharWindows, VOCAB
from wordfusion.data import tensor_to_rgb
from wordfusion.textimage_vae import TextImageVAE

S = 64          # image side (cfg.image_size)


def load_model():
    ckpt = "checkpoints/textimage_vae.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing {ckpt} -- train the VAE first")
    m = TextImageVAE(cfg, vocab=VOCAB)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    return m


def quantize(img):
    return ((img + 1.0) * 127.5).round().clamp(0, 255) / 127.5 - 1.0


def make_mask(kind, n, gen):
    """(n,1,S,S) tensor: 1 = keep, 0 = erased."""
    m = torch.ones(n, 1, S, S)
    h = S // 2
    if kind == "left":      m[:, :, :, :h] = 0
    elif kind == "right":   m[:, :, :, h:] = 0
    elif kind == "top":     m[:, :, :h, :] = 0
    elif kind == "center":  m[:, :, S // 4:3 * S // 4, S // 4:3 * S // 4] = 0
    elif kind.startswith("rand"):
        p = float(kind.split(":")[1])                    # fraction erased
        m = (torch.rand(n, 1, S, S, generator=gen) > p).float()
    return m


def heatmap(vec, target=192):
    """per-text-position accuracy in [0,1] -> (target,target,3) green=good, red=bad.
    The vector length is the sequence length (not the latent size); it is folded
    into a near-square grid just for viewing -- positions are NOT image-aligned."""
    v = np.asarray(vec).reshape(-1)
    side = int(np.ceil(np.sqrt(len(v))))
    g = np.zeros(side * side); g[:len(v)] = v
    g = g.reshape(side, side)
    rgb = np.stack([(1 - g) * 255, g * 255, np.full_like(g, 40)], -1).astype(np.uint8)
    scale = max(1, target // side)
    rgb = np.repeat(np.repeat(rgb, scale, 0), scale, 1)
    out = np.full((target, target, 3), 255, np.uint8)
    out[:rgb.shape[0], :rgb.shape[1]] = rgb[:target, :target]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="texts to average over")
    ap.add_argument("--out", default="samples/mask")
    args = ap.parse_args()

    model = load_model()
    hw = cfg.ti_latent_hw
    # full-length content windows so every position is real (no PAD)
    ds = CharWindows(cfg.corpus_dir, cfg.text_max_len, epoch_size=args.n,
                     random_frac=0.0, var_len=False, seed=7)
    tok = torch.stack([ds[i] for i in range(args.n)]).reshape(args.n, -1)  # (n, max_len)

    with torch.no_grad():
        mean_t, _ = model.encode_text(tok)
        image = model.decode_image(mean_t)                                # clean code-image
    gen = torch.Generator().manual_seed(0)
    os.makedirs(args.out, exist_ok=True)

    def recover(mask):
        with torch.no_grad():
            img_m = quantize(image * mask)
            mean_i, _ = model.encode_image(img_m)
            pred = model.decode_text(mean_i).argmax(-1)                   # (n,256)
        correct = (pred == tok).float()                                  # (n,256)
        return correct.mean(0).numpy(), float(correct.mean())            # per-pos, overall

    masks = ["none", "rand:0.1", "rand:0.25", "rand:0.5", "rand:0.75",
             "left", "right", "top", "center"]
    print(f"averaging over {args.n} texts\n")
    print(f"  {'mask':<12} {'via-img acc':>11}")
    print("  " + "-" * 25)
    rows = {}
    for k in masks:
        mask = torch.ones(args.n, 1, S, S) if k == "none" else make_mask(k, args.n, gen)
        per_pos, overall = recover(mask)
        rows[k] = (per_pos, overall, mask)
        print(f"  {k:<12} {overall:>11.3f}")

    # visuals: for each spatial mask, [masked example image | accuracy heatmap]
    panels = []
    for k in ["none", "left", "right", "top", "center"]:
        per_pos, _, mask = rows[k]
        ex = tensor_to_rgb((image[0:1] * mask[0:1])[0])          # masked example
        ex = np.repeat(np.repeat(ex, 3, 0), 3, 1)                 # 64->192
        hm = heatmap(per_pos)                                     # 192
        gap = np.full((ex.shape[0], 8, 3), 255, np.uint8)
        panels.append(np.concatenate([ex, gap, hm], 1))
    montage = np.concatenate(
        [np.concatenate([np.full((10, panels[0].shape[1], 3), 255, np.uint8), p], 0)
         for p in panels], 0)
    Image.fromarray(montage).save(os.path.join(args.out, "mask_report.png"))
    print(f"\nleft column = masked image, right column = per-position accuracy "
          f"(green=recovered, red=lost); positions folded to a square, NOT image-aligned.")
    print(f"montage -> {args.out}/mask_report.png")
    print("with the Perceiver encoder text is superposed (non-localized), so expect the")
    print("heatmap to dim ~evenly wherever you mask -- and graceful decay vs erase fraction.")


if __name__ == "__main__":
    main()
