"""How much text can we pack into the fixed latent/image? A capacity sweep.

The latent is fixed (ti_latent_ch * ti_latent_hw^2 floats, e.g. 4*16*16 = 1024).
We feed content of increasing length L into it and measure how much survives the
round-trip -- separately for:
  - REAL text  (redundant, so it packs more before breaking), and
  - RANDOM ASCII chars (max entropy = the true channel capacity of the latent).

The length where accuracy falls off the cliff is where superposition saturates:
the model is trying to store more independent symbols than the latent can hold.

Reported per length L:
  acc_z  / acc_img : per-character accuracy (via z, and through the 8-bit image)
  exact_img        : fraction of sequences reconstructed EXACTLY through the image
  bits/float       : L*log2(alphabet) / (#latent floats) -- how hard you push the latent

CPU only; does not touch the GPU. Writes a capacity curve PNG.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw

from wordfusion.config import DEFAULT as cfg
from wordfusion.textio import VOCAB, PAD_ID, N_CHARS, CharWindows, tokens_to_text
from wordfusion.textimage_vae import TextImageVAE


def load_model():
    ckpt = "checkpoints/textimage_vae.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing {ckpt} -- train the VAE first")
    m = TextImageVAE(cfg, vocab=VOCAB)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    return m


def make_batch(source, corpus, L, n, rng):
    toks = np.full((n, cfg.text_max_len), PAD_ID, dtype=np.int64)
    for i in range(n):
        if source == "random":
            c = rng.integers(0, N_CHARS, size=L)
        else:
            s = int(rng.integers(0, max(1, len(corpus) - L)))
            c = corpus[s:s + L]
        toks[i, :len(c)] = c
    return torch.from_numpy(toks).long()


def evaluate(model, tok):
    with torch.no_grad():
        out = model(tok, quantize=cfg.ti_quantize, img_noise=cfg.ti_img_noise)
    content = tok != PAD_ID
    n_content = content.sum().clamp(min=1)
    res = {}
    for key, name in [("logits_t", "z"), ("logits_i", "img")]:
        pred = out[key].argmax(-1)                                  # (n, max_len)
        acc = float(((pred == tok) & content).sum().item()) / int(n_content)
        exact = np.mean([tokens_to_text(tok[i].numpy()) == tokens_to_text(pred[i].numpy())
                         for i in range(tok.size(0))])
        res[name] = (acc, float(exact))
    return res


def plot(lengths, curves, path, latent_floats):
    W, H, m = 560, 340, 46
    im = Image.new("RGB", (W, H), (255, 255, 255)); d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = m, H - m, W - m, m
    d.rectangle([x0, y1, x1, y0], outline=(0, 0, 0))
    for f in (0, .25, .5, .75, 1):                                  # y grid + labels
        y = y0 + (y1 - y0) * f
        d.line([x0, y, x1, y], fill=(230, 230, 230)); d.text((6, y - 5), f"{f:.2f}", fill=(0, 0, 0))
    lx = np.log2(np.array(lengths, float))
    def px(v): return x0 + (x1 - x0) * (v - lx[0]) / (lx[-1] - lx[0] + 1e-9)
    for L in lengths:
        d.text((px(np.log2(L)) - 6, y0 + 6), str(L), fill=(0, 0, 0))
    colors = {"real via-img": (30, 120, 220), "random via-img": (220, 60, 60),
              "real via-z": (140, 190, 250), "random via-z": (250, 150, 150)}
    ly = 8
    for label, ys in curves.items():
        col = colors.get(label, (0, 0, 0))
        pts = [(px(np.log2(L)), y0 + (y1 - y0) * a) for L, a in zip(lengths, ys)]
        d.line(pts, fill=col, width=2)
        for p in pts:
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill=col)
        d.text((x1 - 150, y1 + ly), label, fill=col); ly += 12
    d.text((m, 6), f"capacity: accuracy vs length ({latent_floats} latent floats)", fill=(0, 0, 0))
    d.text((W // 2 - 40, H - 16), "content length (log2)", fill=(0, 0, 0))
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="sequences per length "
                    "(kept low: long max_len self-attention on CPU is heavy)")
    ap.add_argument("--out", default="samples/capacity")
    args = ap.parse_args()

    model = load_model()
    corpus = CharWindows(cfg.corpus_dir, cfg.text_max_len, epoch_size=1).data
    rng = np.random.default_rng(0)
    latent_floats = cfg.ti_latent_ch * cfg.ti_latent_hw ** 2
    bits_per_sym = np.log2(N_CHARS)                     # ~6.6 bits per ASCII char
    lengths = [L for L in (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
               if L <= cfg.text_max_len]
    if cfg.text_max_len not in lengths:
        lengths.append(cfg.text_max_len)

    os.makedirs(args.out, exist_ok=True)
    print(f"latent = {latent_floats} floats  |  max_len = {cfg.text_max_len}  |  "
          f"{args.n} seqs/length\n")
    header = f"  {'len':>5} {'bits/flt':>8} | {'real acc_z':>10} {'real acc_img':>12} " \
             f"{'real exact':>10} | {'rand acc_z':>10} {'rand acc_img':>12} {'rand exact':>10}"
    print(header); print("  " + "-" * (len(header) - 2))
    curves = {"real via-img": [], "random via-img": [],
              "real via-z": [], "random via-z": []}
    for L in lengths:
        r = evaluate(model, make_batch("real", corpus, L, args.n, rng))
        g = evaluate(model, make_batch("random", corpus, L, args.n, rng))
        curves["real via-z"].append(r["z"][0]);     curves["real via-img"].append(r["img"][0])
        curves["random via-z"].append(g["z"][0]);   curves["random via-img"].append(g["img"][0])
        print(f"  {L:>5} {L*bits_per_sym/latent_floats:>8.2f} | {r['z'][0]:>10.3f} {r['img'][0]:>12.3f} "
              f"{r['img'][1]:>10.3f} | {g['z'][0]:>10.3f} {g['img'][0]:>12.3f} {g['img'][1]:>10.3f}")

    plot(lengths, curves, os.path.join(args.out, "capacity.png"), latent_floats)
    print(f"\ncurve -> {args.out}/capacity.png")
    print("real >> random at a given length = the model exploits text redundancy.")
    print("the length where accuracy drops = where the latent's superposition saturates.")


if __name__ == "__main__":
    main()
