"""Smoke-test diffusion UNet + sampler shapes on random data (no training)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from wordfusion.config import DEFAULT as cfg
from wordfusion.diffusion import UNet, GaussianDiffusion

device = cfg.device if torch.cuda.is_available() else "cpu"
lh, lw = cfg.latent_hw
unet = UNet(cfg.latent_ch, base_ch=cfg.unet_base_ch, ch_mult=cfg.unet_ch_mult).to(device)
print("unet params:", sum(p.numel() for p in unet.parameters()) / 1e6, "M")

x = torch.randn(4, cfg.latent_ch, lh, lw, device=device)
t = torch.randint(0, 1000, (4,), device=device)
out = unet(x, t)
print("forward:", tuple(x.shape), "->", tuple(out.shape), "OK" if out.shape == x.shape else "SHAPE MISMATCH")

diff = GaussianDiffusion(timesteps=50, device=device)  # short for speed
loss = diff.p_losses(unet, x)
print("p_losses:", float(loss))

z = diff.sample(unet, (2, cfg.latent_ch, lh, lw))
print("uncond sample:", tuple(z.shape))

# inpainting path
gx = torch.randn(2, cfg.latent_ch, lh, lw, device=device)
gm = torch.zeros(2, 1, lh, lw, device=device); gm[:, :, :lh // 2, :] = 1
z2 = diff.sample(unet, (2, cfg.latent_ch, lh, lw), guidance_x=gx, guidance_mask=gm)
pinned_ok = torch.allclose(z2[:, :, :lh // 2, :], gx[:, :, :lh // 2, :])
print("inpaint sample:", tuple(z2.shape), "pinned region matches:", pinned_ok)
print("ALL OK")
