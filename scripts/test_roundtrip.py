"""Round-trip arbitrary (non-Shakespeare) text through the trained VAE.

For each input text it runs the full loop
    text -> z -> image -> z -> text
and reports character accuracy on the DIRECT path (via z) and the through-the-
image path (via img), measured over the real (non-PAD) characters only.

The model was trained only on Shakespeare, so out-of-distribution text (modern
prose, code, numbers, random chars) should reconstruct worse -- that gap is the
overfitting that random-character training would fix.

CPU only; does not touch the GPU. Pass your own text with --text "..." or a file
of one-text-per-line with --file path. Representation is character-level over a
fixed ASCII alphabet (newline + ' '..'~'); characters outside it fold to '?'.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image

from wordfusion.config import DEFAULT as cfg
from wordfusion.textio import VOCAB, PAD_ID, text_to_tokens, tokens_to_text
from wordfusion.data import tensor_to_rgb
from wordfusion.textimage_vae import TextImageVAE

BUILTIN = {
    "pangram":  "The quick brown fox jumps over the lazy dog. Pack my box with five dozen jugs.",
    "code":     "def add(a, b):\n    return a + b\n\nprint(add(2, 3))  # 5",
    "numbers":  "Order #48213 shipped 2024-11-05 for $1,299.99 to ZIP 90210, qty x7.",
    "casual":   "omg this is literally the best thing ever lol :) cant wait!!! 100%",
    "url":      "https://example.com/path?q=hello&page=2#section-4 (see docs)",
    "unicode":  "café ☃ 日本語 naïve €50 — emoji test 🙂",
}


def load_model():
    ckpt = "checkpoints/textimage_vae.pt"
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing {ckpt} -- train the VAE first")
    m = TextImageVAE(cfg, vocab=VOCAB)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")["model"])
    m.eval()
    return m


def roundtrip(model, text):
    tok = torch.from_numpy(text_to_tokens(text, cfg.text_max_len)).long()[None]  # (1, T)
    with torch.no_grad():
        out = model(tok, quantize=cfg.ti_quantize, img_noise=cfg.ti_img_noise)
    pred_t = out["logits_t"].argmax(-1)[0]                 # (T,)
    pred_i = out["logits_i"].argmax(-1)[0]
    tok0 = tok[0]
    content = tok0 != PAD_ID                               # real bytes only
    n = int(content.sum().item()) or 1
    acc_t = float(((pred_t == tok0) & content).sum().item()) / n
    acc_i = float(((pred_i == tok0) & content).sum().item()) / n
    img = tensor_to_rgb(out["image"][0])
    return acc_t, acc_i, tokens_to_text(pred_t.numpy()), tokens_to_text(pred_i.numpy()), img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None, help="a single text to round-trip")
    ap.add_argument("--file", default=None, help="file with one text per line")
    ap.add_argument("--out", default="samples/roundtrip", help="dir for code-images")
    args = ap.parse_args()

    if args.text is not None:
        items = [("input", args.text)]
    elif args.file:
        with open(args.file, encoding="utf-8") as f:
            items = [(f"line{i}", ln.rstrip("\n")) for i, ln in enumerate(f) if ln.strip()]
    else:
        items = list(BUILTIN.items())

    model = load_model()
    os.makedirs(args.out, exist_ok=True)
    print(f"{'label':<10} {'acc_z':>6} {'acc_img':>7}   reconstruction (via img)")
    print("-" * 78)
    imgs = []
    for label, text in items:
        at, ai, rt, ri, img = roundtrip(model, text)
        imgs.append(img)
        Image.fromarray(np.repeat(np.repeat(img, 4, 0), 4, 1)).save(
            os.path.join(args.out, f"{label}.png"))
        print(f"{label:<10} {at:>6.3f} {ai:>7.3f}   {ri[:48]!r}")
    print(f"\ncode-images -> {args.out}/")
    print("low acc on non-Shakespeare = the model overfit to Shakespeare character statistics.")


if __name__ == "__main__":
    main()
