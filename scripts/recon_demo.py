"""Milestone 1 demo: text -> image -> VAE compress -> image -> text round-trip.

Shows that the codec + neural compression preserve readable text, and saves a
side-by-side (original | reconstructed) image strip.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from wordfusion.config import DEFAULT as cfg
from wordfusion.data import ShakespeareWindows, render_rgb, nearest_char_idx
from wordfusion.vae import VAE
from wordfusion.codec import idx_to_text, idx_to_image, upscale

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--ckpt", type=str, default="checkpoints/vae.pt")
args = ap.parse_args()

device = cfg.device if torch.cuda.is_available() else "cpu"
vae = VAE(cfg).to(device).eval()
vae.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

ds = ShakespeareWindows(cfg.corpus_path, cfg.img_h, cfg.img_w, epoch_size=args.n, seed=7)
idx = torch.stack([ds[i] for i in range(args.n)])          # (n, H, W) real windows

with torch.no_grad():
    recon = vae(render_rgb(idx, device))[0]                # decoded RGB
    pred = nearest_char_idx(recon.float()).cpu()

rows = []
total = 0.0
print("=" * 70)
for i in range(args.n):
    orig = idx_to_text(idx[i].numpy())
    rec = idx_to_text(pred[i].numpy())
    acc = (pred[i] == idx[i]).float().mean().item()
    total += acc
    print(f"[{i}] char-acc {acc:.3f}")
    print(f"  in : {orig[:64]!r}")
    print(f"  out: {rec[:64]!r}")
    # build a side-by-side strip: original (left) | reconstruction (right)
    o = upscale(idx_to_image(idx[i].numpy()), 12)
    r = upscale(idx_to_image(pred[i].numpy()), 12)
    sep = np.full((o.shape[0], 6, 3), 255, np.uint8)
    rows.append(np.concatenate([o, sep, r], axis=1))
print("=" * 70)
print(f"mean char-acc over {args.n} windows: {total/args.n:.3f}")

gap = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
strip = rows[0]
for r in rows[1:]:
    strip = np.concatenate([strip, gap, r], axis=0)
os.makedirs("samples", exist_ok=True)
Image.fromarray(strip).save("samples/recon_demo.png")
print("saved samples/recon_demo.png  (left=original | right=VAE reconstruction)")
