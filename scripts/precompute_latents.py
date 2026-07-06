"""Pre-embed the corpus: text windows -> images -> frozen-VAE latents, cached to disk.

Diffusion then trains directly on these cached latents (no VAE forward per step).
We store the VAE's posterior MEAN per window (deterministic, stable targets) plus the
latent-normalization scale (1/std) so training can rescale to ~unit variance.
"""
import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from wordfusion.config import DEFAULT as cfg
from wordfusion.data import ShakespeareWindows, render_rgb
from wordfusion.vae import VAE

ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=16, help="chars between window starts")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--vae_ckpt", type=str, default="checkpoints/vae.pt")
ap.add_argument("--out", type=str, default="data/latents.pt")
args = ap.parse_args()

device = cfg.device if torch.cuda.is_available() else "cpu"

vae = VAE(cfg).to(device).eval()
vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])
for p in vae.parameters():
    p.requires_grad_(False)

# corpus -> index codes (reuse the dataset's vectorized converter)
ds = ShakespeareWindows(cfg.corpus_path, cfg.img_h, cfg.img_w, epoch_size=1, seed=0)
codes = ds.codes
win = cfg.capacity
starts = list(range(0, len(codes) - win, args.stride))
print(f"corpus {len(codes)} chars -> {len(starts)} windows (stride {args.stride}, win {win})")

# build all index windows as a big array, then batch-encode
windows = np.stack([codes[s:s + win].reshape(cfg.img_h, cfg.img_w) for s in starts])
windows = torch.from_numpy(windows).long()

lat = []
with torch.no_grad():
    for i in range(0, len(windows), args.batch):
        idx = windows[i:i + args.batch]
        mean, _ = vae.encode(render_rgb(idx, device))
        lat.append(mean.cpu())
        if (i // args.batch) % 20 == 0:
            print(f"  encoded {i + len(idx)}/{len(windows)}")
latents = torch.cat(lat)                      # (N, latent_ch, h, w) float32
scale = 1.0 / latents.std().item()            # LDM latent normalization

os.makedirs(os.path.dirname(args.out), exist_ok=True)
torch.save({
    "latents": latents.half(),                # store fp16 to halve disk/RAM
    "scale": scale,
    "cfg": {"latent_ch": cfg.latent_ch, "latent_hw": cfg.latent_hw,
            "img_h": cfg.img_h, "img_w": cfg.img_w},
}, args.out)

mb = latents.numel() * 2 / 1e6
print(f"saved {args.out}: latents {tuple(latents.shape)} (~{mb:.0f} MB fp16), scale {scale:.4f}")
