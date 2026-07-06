"""Latent diffusion for WORDFUSION: a small time-conditioned UNet + DDPM.

The UNet denoises the VAE latent (latent_ch, h, w). This is the generative model
-- "the LLM." It models p(latent), i.e. the distribution of encoded text passages,
and denoises an entire passage at once rather than left-to-right.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- building blocks -------------------------------------------------------

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embedding, (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn(ch):
    return nn.GroupNorm(min(32, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_dim):
        super().__init__()
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(temb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = _gn(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        attn = torch.softmax(q @ k / math.sqrt(c), dim=-1)
        out = (attn @ v).permute(0, 2, 1).reshape(b, c, h, w)
        return x + self.proj(out)


class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, latent_ch, base_ch=128, ch_mult=(1, 2, 2), attn_at=(1, 2)):
        super().__init__()
        temb_dim = base_ch * 4
        self.temb = nn.Sequential(nn.Linear(base_ch, temb_dim), nn.SiLU(),
                                  nn.Linear(temb_dim, temb_dim))
        self.base_ch = base_ch
        self.in_conv = nn.Conv2d(latent_ch, base_ch, 3, padding=1)

        chs = [base_ch]
        ch = base_ch
        self.downs = nn.ModuleList()
        for level, mult in enumerate(ch_mult):
            out = base_ch * mult
            self.downs.append(ResBlock(ch, out, temb_dim)); ch = out; chs.append(ch)
            if level in attn_at:
                self.downs.append(AttnBlock(ch))
            if level < len(ch_mult) - 1:
                self.downs.append(Down(ch)); chs.append(ch)

        self.mid1 = ResBlock(ch, ch, temb_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, temb_dim)

        self.ups = nn.ModuleList()
        for level, mult in reversed(list(enumerate(ch_mult))):
            out = base_ch * mult
            for _ in range(2):
                self.ups.append(ResBlock(ch + chs.pop(), out, temb_dim)); ch = out
            if level in attn_at:
                self.ups.append(AttnBlock(ch))
            if level > 0:
                self.ups.append(Up(ch))

        self.out_norm = _gn(ch)
        self.out_conv = nn.Conv2d(ch, latent_ch, 3, padding=1)

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.base_ch))
        h = self.in_conv(x)
        skips = [h]
        for m in self.downs:
            h = m(h, temb) if isinstance(m, ResBlock) else m(h)
            if isinstance(m, (ResBlock, Down)):
                skips.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for m in self.ups:
            if isinstance(m, ResBlock):
                h = m(torch.cat([h, skips.pop()], dim=1), temb)
            else:
                h = m(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# --- gaussian diffusion ----------------------------------------------------

class GaussianDiffusion:
    """Standard DDPM with a cosine beta schedule, epsilon-prediction."""

    def __init__(self, timesteps=1000, device="cuda"):
        self.T = timesteps
        betas = self._cosine_betas(timesteps).to(device)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_acp = torch.sqrt(self.alphas_cumprod)
        self.sqrt_om_acp = torch.sqrt(1 - self.alphas_cumprod)
        self.device = device

    @staticmethod
    def _cosine_betas(T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T + s) / (1 + s)) * math.pi / 2) ** 2
        acp = f / f[0]
        betas = 1 - (acp[1:] / acp[:-1])
        return betas.clamp(1e-8, 0.999).float()

    def q_sample(self, x0, t, noise):
        return self.sqrt_acp[t][:, None, None, None] * x0 + \
               self.sqrt_om_acp[t][:, None, None, None] * noise

    def p_losses(self, model, x0):
        b = x0.shape[0]
        t = torch.randint(0, self.T, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = model(xt, t)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, model, shape, guidance_x=None, guidance_mask=None):
        """Ancestral DDPM sampling. Optional inpainting via (guidance_x, mask):
        pixels where mask==1 are pinned to guidance_x at every step (=prompting)."""
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            if guidance_x is not None:
                known = self.q_sample(guidance_x, t, torch.randn_like(x))
                x = torch.where(guidance_mask.bool(), known, x)
            eps = model(x, t)
            acp = self.alphas_cumprod[i]
            beta = self.betas[i]
            alpha = 1 - beta
            mean = (x - beta / self.sqrt_om_acp[i] * eps) / torch.sqrt(alpha)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        if guidance_x is not None:
            x = torch.where(guidance_mask.bool(), guidance_x, x)
        return x
