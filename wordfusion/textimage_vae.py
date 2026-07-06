"""Bidirectional text<->image VAE for WORDFUSION (learned codec, no palette).

Unlike the palette codec + image-only VAE, this model *learns* the whole
text <-> image <-> latent mapping end to end. There is ONE shared latent

    z : (ti_latent_ch, hw, hw)          e.g. (4, 16, 16)   <- diffusion lives here

and an image side

    z' : (3, image_size, image_size)    e.g. (3, 64, 64)   <- an actual RGB image

Four learned pieces, all hung off the single shared z:

    TextEncoder :  tokens (T,)              -> z   (mean, logvar)
    ImageDecoder:  z                        -> image (3, S, S)
    ImageEncoder:  image (3, S, S)          -> z   (mean, logvar)
    TextDecoder :  z                        -> token logits (T, vocab)

The system is BIDIRECTIONAL and trained on text only. From a text batch we run
the full loop  text -> z -> image -> z -> text  so that *both* entry points work:
    - text -> z -> text        (direct reconstruction)
    - image -> z -> text       (read text back out of the generated image)
Nothing pulls the image toward looking natural, so it becomes an arbitrary
learned "code image". Sampleability is NOT this model's job -- the latent
diffusion stage models p(z); here KL is only lightly applied.

The image encoder/decoder reuse the conv stacks from `vae.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import Encoder, Decoder
from .textio import PAD_ID


# --- text heads (global: attend over the whole passage) --------------------

def quantize_ste(x):
    """Round an image in [-1, 1] to real 8-bit pixels, straight-through.

    Forward pass sees genuine uint8-quantized values (what a stored image is);
    the gradient passes through unchanged so the decoder still trains.
    """
    q = ((x + 1.0) * 127.5).round().clamp(0, 255) / 127.5 - 1.0
    return x + (q - x).detach()


def _transformer(dim: int, layers: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=dim, nhead=heads, dim_feedforward=dim * 4,
        dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
    )
    # enable_nested_tensor=False silences the norm_first UserWarning (which
    # PowerShell otherwise wraps in a scary red NativeCommandError box).
    return nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)


class TextEncoder(nn.Module):
    """Variable-length tokens (B, T) -> fixed latent moments (B, latent_ch, hw, hw).

    Perceiver-style: the T tokens are processed, then a FIXED set of hw*hw learned
    latent queries cross-attend into them and pool the whole (any-length) sequence
    into the fixed 16x16 latent. Length is decoupled from latent size, so text
    packs into the latent by *superposition* rather than one-token-per-cell.
    """

    def __init__(self, vocab, dim, layers, heads, latent_ch, hw, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.encoder = _transformer(dim, layers, heads)
        self.queries = nn.Parameter(torch.randn(hw * hw, dim) * 0.02)
        self.q_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.latent = _transformer(dim, max(1, layers // 2), heads)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 2 * latent_ch)
        self.latent_ch, self.hw = latent_ch, hw

    def forward(self, tokens, pad_mask):             # tokens (B,T); pad_mask (B,T) True=pad
        B, T = tokens.shape
        pos = torch.arange(T, device=tokens.device)
        h = self.tok(tokens) + self.pos(pos)[None]
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        q = self.queries.expand(B, -1, -1)           # (B, hw*hw, dim)
        attn, _ = self.cross(self.q_norm(q), h, h, key_padding_mask=pad_mask)
        a = q + attn                                 # residual: direct gradient to queries
        a = self.latent(a)
        m = self.head(self.norm(a))                  # (B, hw*hw, 2*latent_ch)
        m = m.transpose(1, 2).reshape(B, 2 * self.latent_ch, self.hw, self.hw)
        mean, logvar = m.chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)


class TextDecoder(nn.Module):
    """Fixed latent z (B, latent_ch, hw, hw) -> variable-length logits (B, L, vocab).

    Perceiver-style: L positional output queries cross-attend into the hw*hw latent
    cells and read out one token per position. L defaults to max_len; PAD marks the
    end of the real content.
    """

    def __init__(self, vocab, dim, layers, heads, latent_ch, hw, max_len):
        super().__init__()
        self.proj = nn.Linear(latent_ch, dim)
        self.latent_pos = nn.Parameter(torch.randn(hw * hw, dim) * 0.02)
        self.out_pos = nn.Embedding(max_len, dim)
        self.q_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.decoder = _transformer(dim, layers, heads)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.max_len = max_len

    def forward(self, z, out_len=None):              # z (B, C, hw, hw)
        B, C, H, W = z.shape
        L = out_len or self.max_len
        lat = z.reshape(B, C, H * W).transpose(1, 2)         # (B, hw*hw, C)
        lat = self.proj(lat) + self.latent_pos[None]         # (B, hw*hw, dim)
        pos = torch.arange(L, device=z.device)
        q = self.out_pos(pos)[None].expand(B, -1, -1)        # (B, L, dim)
        attn, _ = self.cross(self.q_norm(q), lat, lat)       # (B, L, dim)
        a = q + attn                                         # residual
        a = self.decoder(a)
        return self.head(self.norm(a))                       # (B, L, vocab)


# --- the bidirectional VAE --------------------------------------------------

class TextImageVAE(nn.Module):
    def __init__(self, cfg, vocab: int):
        super().__init__()
        hw = cfg.ti_latent_hw
        n_down = int(round(math.log2(cfg.ti_downsample)))
        assert 2 ** n_down == cfg.ti_downsample, "ti_downsample must be a power of 2"

        # image side reuses the conv autoencoder from vae.py
        self.image_encoder = Encoder(cfg.ti_base_ch, cfg.ti_latent_ch, n_down)
        self.image_decoder = Decoder(cfg.ti_base_ch, cfg.ti_latent_ch, n_down)
        # text side: Perceiver-style variable-length heads
        self.text_encoder = TextEncoder(vocab, cfg.ti_text_dim, cfg.ti_text_layers,
                                        cfg.ti_text_heads, cfg.ti_latent_ch, hw,
                                        cfg.text_max_len)
        self.text_decoder = TextDecoder(vocab, cfg.ti_text_dim, cfg.ti_text_layers,
                                        cfg.ti_text_heads, cfg.ti_latent_ch, hw,
                                        cfg.text_max_len)
        self.latent_ch, self.hw, self.vocab = cfg.ti_latent_ch, hw, vocab

    @staticmethod
    def reparameterize(mean, logvar):
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)

    # -- the four primitive mappings --
    def encode_text(self, tokens):
        return self.text_encoder(tokens, tokens.eq(PAD_ID))
    def encode_image(self, image):
        mean, logvar = self.image_encoder(image)
        return mean, logvar.clamp(-30.0, 20.0)
    def decode_text(self, z, out_len=None):  return self.text_decoder(z, out_len)
    def decode_image(self, z):      return self.image_decoder(z)  # (B,3,S,S) in [-1,1]

    def forward(self, tokens, quantize=False, img_noise=0.0):
        """Full bidirectional loop from a text batch; returns everything the
        loss needs. All four paths share the single latent z.

        `quantize`/`img_noise` perturb the image *before* it is re-encoded, so
        the text must survive a real (8-bit, optionally noised) image, not just
        a clean float tensor.
        """
        mean_t, logvar_t = self.encode_text(tokens)
        z_t = self.reparameterize(mean_t, logvar_t)

        logits_t = self.decode_text(z_t)              # text -> z -> text
        image = self.decode_image(z_t)                # z -> image (arbitrary)

        # what actually gets re-encoded: the image as it would be stored/noised
        img_in = image
        if quantize:
            img_in = quantize_ste(img_in)
        if img_noise > 0:
            img_in = img_in + img_noise * torch.randn_like(img_in)

        mean_i, logvar_i = self.encode_image(img_in)  # image -> z
        z_i = self.reparameterize(mean_i, logvar_i)
        logits_i = self.decode_text(z_i)              # image -> z -> text
        image_rec = self.decode_image(z_i)            # image -> z -> image

        return {
            "logits_t": logits_t, "logits_i": logits_i,
            "image": image, "image_rec": image_rec,
            "z_t": z_t, "mean_t": mean_t, "logvar_t": logvar_t,
            "mean_i": mean_i,
        }


def ti_loss(out, tokens, cfg):
    """Bidirectional loss. Text is the only supervised target; the image is
    supervised only for *self-consistency*, never toward natural images.

    The text CE is masked to the real content chars + one stop token (the first
    PAD), so trailing PAD (up to ~98% of a long window) can't dominate the loss
    and collapse the model to predicting blanks.
    """
    V = out["logits_t"].size(-1)
    tgt = tokens.reshape(-1)
    B, T = tokens.shape
    lens = (tokens != PAD_ID).sum(1)                                # content length / row
    ar = torch.arange(T, device=tokens.device)
    loss_mask = (ar[None, :] <= lens[:, None]).reshape(-1).float()  # content + first PAD (stop)
    denom = loss_mask.sum().clamp(min=1)

    def masked_ce(logits):
        ce = F.cross_entropy(logits.reshape(-1, V), tgt, reduction="none")
        return (ce * loss_mask).sum() / denom

    ce_t = masked_ce(out["logits_t"])                               # text->z->text
    ce_i = masked_ce(out["logits_i"])                              # image->z->text

    # KL in fp32: logvar.exp() overflows fp16 (exp(20) >> 65504) under AMP.
    mean_t, logvar_t = out["mean_t"].float(), out["logvar_t"].float()
    kl = -0.5 * torch.mean(1 + logvar_t - mean_t.pow(2) - logvar_t.exp())

    # image must faithfully carry z: re-encoding the generated image returns z_t,
    # and re-decoding that returns the same image.
    cycle = F.mse_loss(out["mean_i"], out["z_t"].detach())
    img_rec = F.mse_loss(out["image_rec"], out["image"].detach())

    total = (ce_t + ce_i
             + cfg.ti_lambda_cycle * cycle
             + cfg.ti_lambda_img * img_rec
             + cfg.ti_beta_kl * kl)

    with torch.no_grad():
        content = tokens != PAD_ID                                  # ignore PAD in acc
        n = content.sum().clamp(min=1)
        acc_t = ((out["logits_t"].argmax(-1) == tokens) & content).sum() / n
        acc_i = ((out["logits_i"].argmax(-1) == tokens) & content).sum() / n
    metrics = {"ce_t": ce_t.detach(), "ce_i": ce_i.detach(), "kl": kl.detach(),
               "cycle": cycle.detach(), "img_rec": img_rec.detach(),
               "acc_t": acc_t, "acc_i": acc_i}
    return total, metrics
