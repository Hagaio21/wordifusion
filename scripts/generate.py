"""Generate text with WORDFUSION: sample latents -> VAE decode -> nearest-color text."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from wordfusion.config import DEFAULT as cfg
from wordfusion.vae import VAE
from wordfusion.diffusion import UNet, GaussianDiffusion
from wordfusion.data import tensor_to_rgb, rgb_to_tensor, nearest_char_idx
from wordfusion.codec import idx_to_text, text_to_idx, idx_to_image, upscale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--prompt", type=str, default=None,
                    help="if set, inpaint: fix the prompt's pixels and diffuse the rest")
    ap.add_argument("--vae_ckpt", type=str, default="checkpoints/vae.pt")
    ap.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion.pt")
    ap.add_argument("--out", type=str, default="samples")
    args = ap.parse_args()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    vae = VAE(cfg).to(device).eval()
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])

    d = torch.load(args.diff_ckpt, map_location=device)
    scale = d["scale"]
    unet = UNet(cfg.latent_ch, base_ch=cfg.unet_base_ch, ch_mult=cfg.unet_ch_mult).to(device).eval()
    unet.load_state_dict(d["model"])
    diff = GaussianDiffusion(timesteps=cfg.timesteps, device=device)

    lh, lw = cfg.latent_hw
    shape = (args.n, cfg.latent_ch, lh, lw)

    guidance_x = guidance_mask = None
    if args.prompt:
        # place the prompt in the top-left of the grid; freeze those latent cells
        idx = text_to_idx(args.prompt, cfg.img_h, cfg.img_w)
        rgb = rgb_to_tensor(idx_to_image(idx)).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _ = vae.encode(rgb)
        guidance_x = (mean * scale).repeat(args.n, 1, 1, 1)
        # mask latent rows covered by the prompt (approx: prompt fills first k chars)
        k_chars = min(len(args.prompt), cfg.capacity)
        k_rows_img = int(np.ceil(k_chars / cfg.img_w))
        k_rows_lat = max(1, int(np.ceil(k_rows_img / cfg.downsample)))
        m = torch.zeros(1, 1, lh, lw, device=device)
        m[:, :, :k_rows_lat, :] = 1.0
        guidance_mask = m.repeat(args.n, 1, 1, 1)
        print(f"inpainting prompt over top {k_rows_lat}/{lh} latent rows: {args.prompt!r}")

    with torch.no_grad():
        z = diff.sample(unet, shape, guidance_x=guidance_x, guidance_mask=guidance_mask)
        recon = vae.decode(z / scale)
        idx = nearest_char_idx(recon.float()).cpu()

    print("=" * 60)
    for i in range(args.n):
        text = idx_to_text(idx[i].numpy())
        print(f"[{i}] {text!r}")
        img = idx_to_image(idx[i].numpy())
        Image.fromarray(upscale(img, 16)).save(os.path.join(args.out, f"gen_{i:02d}.png"))
    print("=" * 60)
    print(f"saved {args.n} images to {args.out}/")


if __name__ == "__main__":
    main()
