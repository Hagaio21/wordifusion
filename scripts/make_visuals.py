"""Produce large visual montages of every stage of WORDFUSION."""
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
from wordfusion.diffusion import UNet, GaussianDiffusion
from wordfusion.codec import idx_to_text, idx_to_image, upscale

CELL = 14        # pixels per character block
GAP = 8          # gap within an original|recon pair
PAD = 16         # gap between grid items
BG = 255         # white background

ap = argparse.ArgumentParser()
ap.add_argument("--vae_ckpt", default="checkpoints/vae.pt")
ap.add_argument("--diff_ckpt", default="checkpoints/diffusion.pt")
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--rows", type=int, default=4)
args = ap.parse_args()

device = cfg.device if torch.cuda.is_available() else "cpu"
os.makedirs("samples", exist_ok=True)

vae = VAE(cfg).to(device).eval()
vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])


def block(idx_grid):
    return upscale(idx_to_image(idx_grid), CELL)


def grid_of(items, cols, pad=PAD):
    """Tile a list of equal-size uint8 images into a grid with white padding."""
    h, w = items[0].shape[:2]
    rows = (len(items) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), BG, np.uint8)
    for i, im in enumerate(items):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad); x = pad + c * (w + pad)
        canvas[y:y + h, x:x + w] = im
    return canvas


n = args.rows * args.cols

# ---- 1. reconstruction gallery: original | reconstruction pairs ----
ds = ShakespeareWindows(cfg.corpus_path, cfg.img_h, cfg.img_w, epoch_size=n, seed=3)
idx = torch.stack([ds[i] for i in range(n)])
with torch.no_grad():
    recon = vae(render_rgb(idx, device))[0]
    pred = nearest_char_idx(recon.float()).cpu()

pairs = []
accs = []
for i in range(n):
    a = (pred[i] == idx[i]).float().mean().item(); accs.append(a)
    o = block(idx[i].numpy()); r = block(pred[i].numpy())
    sep = np.full((o.shape[0], GAP, 3), BG, np.uint8)
    pairs.append(np.concatenate([o, sep, r], axis=1))
Image.fromarray(grid_of(pairs, args.cols)).save("samples/gallery_recon.png")
print(f"recon gallery: mean acc {np.mean(accs):.3f}  -> samples/gallery_recon.png")

# ---- 2. generated samples from the diffusion model ----
gen_texts = []
if os.path.exists(args.diff_ckpt):
    d = torch.load(args.diff_ckpt, map_location=device)
    scale = d["scale"]
    unet = UNet(cfg.latent_ch, base_ch=cfg.unet_base_ch, ch_mult=cfg.unet_ch_mult).to(device).eval()
    unet.load_state_dict(d["model"])
    diff = GaussianDiffusion(timesteps=cfg.timesteps, device=device)
    lh, lw = cfg.latent_hw
    with torch.no_grad():
        z = diff.sample(unet, (n, cfg.latent_ch, lh, lw))
        rec = vae.decode(z / scale)
        gidx = nearest_char_idx(rec.float()).cpu()
    gens = [block(gidx[i].numpy()) for i in range(n)]
    Image.fromarray(grid_of(gens, args.cols)).save("samples/gallery_generated.png")
    for i in range(n):
        gen_texts.append(idx_to_text(gidx[i].numpy()))
    print("generated gallery -> samples/gallery_generated.png")
    print("--- generated text samples ---")
    for i, t in enumerate(gen_texts[:6]):
        print(f"[{i}] {t[:70]!r}")
else:
    print("no diffusion checkpoint yet")
