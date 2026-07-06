"""A small KL-regularized VAE for WORDFUSION.

Encoder:  RGB image (3, H, W)  ->  latent (latent_ch, H/d, W/d)
Decoder:  latent               ->  RGB image (3, H, W)

The VAE is a pure IMAGE autoencoder -- it knows nothing about text or the
vocabulary. Turning a reconstructed image back into text is a separate stage
(the codec's nearest-palette lookup, `image_to_idx`). Keeping the two decoders
separate means the VAE just has to reproduce the colors accurately; the palette's
wide spacing lets the nearest-color lookup snap each pixel back to a character.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(ch: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, base_ch: int, latent_ch: int, n_down: int):
        super().__init__()
        self.stem = nn.Conv2d(3, base_ch, 3, padding=1)
        blocks = []
        ch = base_ch
        for _ in range(n_down):
            blocks.append(ResBlock(ch, ch * 2))
            blocks.append(nn.Conv2d(ch * 2, ch * 2, 3, stride=2, padding=1))  # downsample
            ch *= 2
        blocks.append(ResBlock(ch, ch))
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = _gn(ch)
        self.to_moments = nn.Conv2d(ch, 2 * latent_ch, 1)  # mean, logvar

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        h = F.silu(self.out_norm(h))
        mean, logvar = self.to_moments(h).chunk(2, dim=1)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, base_ch: int, latent_ch: int, n_down: int):
        super().__init__()
        ch = base_ch * (2 ** n_down)
        self.stem = nn.Conv2d(latent_ch, ch, 3, padding=1)
        self.in_block = ResBlock(ch, ch)
        blocks = []
        for _ in range(n_down):
            blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            blocks.append(ResBlock(ch, ch // 2))
            ch //= 2
        self.blocks = nn.ModuleList(blocks)
        self.out_norm = _gn(ch)
        self.to_rgb = nn.Conv2d(ch, 3, 1)

    def forward(self, z):
        h = self.in_block(self.stem(z))
        for b in self.blocks:
            h = b(h)
        h = F.silu(self.out_norm(h))
        return torch.tanh(self.to_rgb(h))  # (B, 3, H, W) in [-1, 1]


class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_down = int(round(math.log2(cfg.downsample)))
        assert 2 ** n_down == cfg.downsample, "downsample must be a power of 2"
        self.encoder = Encoder(cfg.vae_base_ch, cfg.latent_ch, n_down)
        self.decoder = Decoder(cfg.vae_base_ch, cfg.latent_ch, n_down)
        self.latent_ch = cfg.latent_ch

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def encode(self, x):
        mean, logvar = self.encoder(x)
        logvar = logvar.clamp(-30.0, 20.0)   # prevent exp(logvar) overflow -> NaN
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar


def vae_loss(recon_img, target_img, mean, logvar, beta: float = 1e-4):
    """Image reconstruction (MSE in [-1,1] RGB space) + KL, weighted by beta."""
    recon = F.mse_loss(recon_img, target_img)
    kl = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()
