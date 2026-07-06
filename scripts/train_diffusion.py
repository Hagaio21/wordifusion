"""Train the latent-diffusion UNet on PRE-EMBEDDED latents, sampling at intervals.

The UNet trains purely over the cached latents (no VAE forward in the loop).
Every --sample_every steps we generate samples, decode them through the frozen
VAE, save the image grid, and print the decoded text -- so you can watch the
model's actual output (image + text) improve as it trains.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from wordfusion.config import DEFAULT as cfg
from wordfusion.vae import VAE
from wordfusion.diffusion import UNet, GaussianDiffusion
from wordfusion.data import nearest_char_idx
from wordfusion.codec import idx_to_image, idx_to_text, upscale


def _font(size=16):
    for p in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\cour.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _wrap(text, width):
    lines = []
    for ln in text.split("\n"):
        if ln == "":
            lines.append("")
        while len(ln) > width:
            lines.append(ln[:width]); ln = ln[width:]
        if ln:
            lines.append(ln)
    return lines or [""]


def _sample_panel(idx_grid, step, i, font, cell=14, wrap=40, bg=(20, 20, 24), fg=(235, 235, 235)):
    """One sample = its colored image (left) + the text decoded from it (right)."""
    img = Image.fromarray(upscale(idx_to_image(idx_grid), cell))     # (16*cell)^2
    ih = img.height
    text = idx_to_text(idx_grid)
    lines = _wrap(text, wrap)
    text_w = wrap * (font.getbbox("M")[2] or 9) + 24
    panel = Image.new("RGB", (img.width + text_w, ih), bg)
    panel.paste(img, (0, 0))
    d = ImageDraw.Draw(panel)
    d.text((img.width + 12, 4), f"sample {i}  (step {step})", font=font, fill=(140, 200, 255))
    lh = font.getbbox("Mg")[3] + 3
    y = 4 + lh + 4
    for ln in lines:
        d.text((img.width + 12, y), ln if ln else "·", font=font,
               fill=fg if ln else (90, 90, 90))
        y += lh
    return np.array(panel)


def sample_and_show(unet, vae, diff, scale, step, n, out_dir):
    unet.eval()
    lh, lw = cfg.latent_hw
    font = _font(16)
    with torch.no_grad():
        z = diff.sample(unet, (n, cfg.latent_ch, lh, lw))
        rec = vae.decode(z / scale)
        gidx = nearest_char_idx(rec.float()).cpu()
    panels = [_sample_panel(gidx[i].numpy(), step, i, font) for i in range(n)]
    w = max(p.shape[1] for p in panels)
    gap = np.full((12, w, 3), 20, np.uint8)
    rows = []
    for p in panels:
        if p.shape[1] < w:
            p = np.concatenate([p, np.full((p.shape[0], w - p.shape[1], 3), 20, np.uint8)], axis=1)
        rows.extend([p, gap])
    canvas = np.concatenate(rows[:-1], axis=0)
    path = os.path.join(out_dir, f"gen_step{step:05d}.png")
    Image.fromarray(canvas).save(path)
    print(f"----- SAMPLES @ step {step}  (image+text: {path}) -----", flush=True)
    for i in range(n):
        print(f"SAMPLE[{step}] {idx_to_text(gidx[i].numpy())[:78]!r}", flush=True)
    print("-" * 60, flush=True)
    unet.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=250)
    ap.add_argument("--n_samples", type=int, default=6)
    ap.add_argument("--latents", type=str, default="data/latents.pt")
    ap.add_argument("--vae_ckpt", type=str, default="checkpoints/vae.pt")
    ap.add_argument("--ckpt", type=str, default="checkpoints/diffusion.pt")
    args = ap.parse_args()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
    os.makedirs("samples", exist_ok=True)

    # frozen VAE (only used to decode samples into images/text)
    vae = VAE(cfg).to(device).eval()
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)["model"])
    for p in vae.parameters():
        p.requires_grad_(False)

    cache = torch.load(args.latents, map_location="cpu")
    scale = cache["scale"]
    latents = cache["latents"].to(device).float()
    N = latents.shape[0]
    print(f"loaded {N} cached latents {tuple(latents.shape[1:])}  scale {scale:.4f}", flush=True)

    unet = UNet(cfg.latent_ch, base_ch=cfg.unet_base_ch, ch_mult=cfg.unet_ch_mult).to(device)
    diff = GaussianDiffusion(timesteps=cfg.timesteps, device=device)
    opt = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    print(f"device={device}  unet={sum(p.numel() for p in unet.parameters())/1e6:.2f}M  batch={args.batch}", flush=True)

    # baseline sample from the untrained model
    sample_and_show(unet, vae, diff, scale, 0, args.n_samples, "samples")

    t0 = time.time(); run = 0.0; seen = 0
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.batch,), device=device, generator=g)
        z0 = latents[idx] * scale
        opt.zero_grad(set_to_none=True)
        loss = diff.p_losses(unet, z0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        opt.step()
        run += loss.item(); seen += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            print(f"step {step:>5}/{args.steps}  loss {run/seen:.4f}  {step*args.batch/dt:.0f} lat/s", flush=True)
            run = 0.0; seen = 0
        if step % args.sample_every == 0:
            torch.save({"model": unet.state_dict(), "scale": scale, "cfg": vars(cfg)}, args.ckpt)
            sample_and_show(unet, vae, diff, scale, step, args.n_samples, "samples")

    torch.save({"model": unet.state_dict(), "scale": scale, "cfg": vars(cfg)}, args.ckpt)
    print(f"saved {args.ckpt}", flush=True)


if __name__ == "__main__":
    main()
