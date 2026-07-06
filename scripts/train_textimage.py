"""Train the bidirectional text<->image VAE (learned codec, no palette).

Text-only corpus. Each step runs the full loop
    text -> z (16x16x4) -> image (64x64x3) -> z -> text
and trains both the direct text round-trip and the image->text round-trip, so
either modality is a valid entry point. The image is never supervised toward
natural images -- it becomes an arbitrary learned code that stores the text.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from tqdm import tqdm

from wordfusion.config import DEFAULT as cfg
from wordfusion.data import tensor_to_rgb
from wordfusion.textio import CharWindows, VOCAB, tokens_to_text
from wordfusion.textimage_vae import TextImageVAE, ti_loss

_FONT = ImageFont.load_default()


def _wrap(s, width):
    s = s.replace("\n", "¶")                     # show newlines as a glyph
    return [s[i:i + width] for i in range(0, max(len(s), 1), width)]


def _fixed(s, cw, k=2):
    """Exactly k wrapped lines (pad with blanks) so every panel is the same height."""
    w = _wrap(s, cw)[:k]
    return w + [""] * (k - len(w))


def _panel(img_u8, orig, via_img, scale=4, bg=245):
    """One sample: the code-image (text->image) with the image->text decode
    and the original rendered underneath, so both directions are visible."""
    im = np.repeat(np.repeat(img_u8, scale, 0), scale, 1)   # (256,256,3)
    w = im.shape[1]
    cw = 42                                            # chars per line at this width
    lines = ([("orig", (90, 90, 90))] + [(l, (90, 90, 90)) for l in _fixed(orig, cw)]
             + [("img>txt", (10, 90, 10))] + [(l, (10, 90, 10)) for l in _fixed(via_img, cw)])
    strip_h = 12 * len(lines) + 6
    strip = Image.new("RGB", (w, strip_h), (bg, bg, bg))
    d = ImageDraw.Draw(strip)
    y = 2
    for text, color in lines:
        d.text((3, y), text, fill=color, font=_FONT); y += 12
    return np.concatenate([im, np.array(strip, np.uint8)], axis=0)


def _tile(panels, cols=4, pad=8, bg=255):
    h, w = panels[0].shape[:2]
    rows = (len(panels) + cols - 1) // cols
    canvas = np.full((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3),
                     bg, np.uint8)
    for i, im in enumerate(panels):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad); x = pad + c * (w + pad)
        canvas[y:y + im.shape[0], x:x + w] = im
    return canvas


def dump_samples(model, tokens, step, out_dir, cfg, device):
    """Write this step's samples to DISK (never the terminal):
      ti_stepNNNNNN.png  -- montage: text->image code-images + image->text decode
      ti_stepNNNNNN.txt  -- the text round-trip (orig / via z / via img) per sample
    Returns (acc_t, acc_i) so the caller can log a single progress line."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        out = model(tokens, quantize=cfg.ti_quantize, img_noise=cfg.ti_img_noise)
        pred_t = out["logits_t"].argmax(-1).cpu()
        pred_i = out["logits_i"].argmax(-1).cpu()
        tok_cpu = tokens.cpu()
        panels, lines = [], [f"step {step}"]
        for i in range(tokens.size(0)):
            img_u8 = tensor_to_rgb(out["image"][i])           # text -> image
            orig = tokens_to_text(tok_cpu[i].numpy())
            via_z = tokens_to_text(pred_t[i].numpy())         # text -> z -> text
            via_img = tokens_to_text(pred_i[i].numpy())       # image -> z -> text
            panels.append(_panel(img_u8, orig, via_img))
            lines += [f"[{i}] orig    : {orig!r}",
                      f"    via z   : {via_z!r}",
                      f"    via img : {via_img!r}", ""]
    Image.fromarray(_tile(panels)).save(os.path.join(out_dir, f"ti_step{step:06d}.png"))
    with open(os.path.join(out_dir, f"ti_step{step:06d}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if was_training:
        model.train()
    at = (pred_t == tok_cpu).float().mean().item()
    ai = (pred_i == tok_cpu).float().mean().item()
    return at, ai


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--sample_every", type=int, default=250)
    ap.add_argument("--sample_dir", type=str, default="samples/ti")
    ap.add_argument("--ckpt", type=str, default="checkpoints/textimage_vae.pt")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--resume", action="store_true",
                    help="continue from --ckpt if it exists (model+optimizer+step)")
    args = ap.parse_args()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    # char-level ASCII, VARIABLE-length windows (short..max_len) + random chars
    hw = cfg.ti_latent_hw
    ds = CharWindows(cfg.corpus_dir, cfg.text_max_len,
                     epoch_size=args.steps * args.batch, min_len=cfg.text_min_len,
                     var_len=True, random_frac=cfg.ti_random_frac, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=(device == "cuda"))

    model = TextImageVAE(cfg, vocab=VOCAB).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": vars(cfg)}, args.ckpt)

    start_step = 0
    if args.resume and os.path.exists(args.ckpt):
        ck = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        start_step = int(ck.get("step", 0))
        print(f"resumed from {args.ckpt} at step {start_step}")

    os.makedirs(args.sample_dir, exist_ok=True)
    # FIXED watch-set: real text of varied lengths + a couple random-char windows,
    # so the samples show short & long, structured & arbitrary content round-tripping.
    real_ds = CharWindows(cfg.corpus_dir, cfg.text_max_len, epoch_size=6,
                          min_len=cfg.text_min_len, var_len=True, random_frac=0.0, seed=123)
    rand_ds = CharWindows(cfg.corpus_dir, cfg.text_max_len, epoch_size=2,
                          min_len=cfg.text_min_len, var_len=True, random_frac=1.0, seed=123)
    sample_tokens = torch.stack([real_ds[i] for i in range(6)] +
                                [rand_ds[i] for i in range(2)]).reshape(8, -1).to(device)

    print(f"device={device}  params={n_params/1e6:.2f}M  "
          f"z={cfg.ti_latent_ch}x{hw}x{hw}  image={cfg.image_size}  "
          f"max_len={cfg.text_max_len}  vocab={VOCAB}(ascii)  "
          f"rand_frac={cfg.ti_random_frac}  batch={args.batch}")

    model.train()
    t0 = time.time()
    run = {}
    seen = 0
    if start_step == 0:
        dump_samples(model, sample_tokens, 0, args.sample_dir, cfg, device)  # starting point
    print(f"samples -> {args.sample_dir}/  (ti_stepNNNNNN.png + .txt every {args.sample_every} steps)")

    bar = tqdm(total=args.steps, initial=start_step, dynamic_ncols=True, smoothing=0.1)
    for local_step, idx in enumerate(dl, 1):
        step = start_step + local_step
        tokens = idx.reshape(idx.size(0), -1).to(device, non_blocking=True)  # (B, T)
        opt.zero_grad(set_to_none=True)
        out = model(tokens, quantize=cfg.ti_quantize, img_noise=cfg.ti_img_noise)
        loss, m = ti_loss(out, tokens, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        for k, v in m.items():
            run[k] = run.get(k, 0.0) + v.item()
        seen += 1

        # live per-step feedback on the bar itself
        bar.update(1)
        bar.set_postfix(acc_t=f"{m['acc_t']:.3f}", acc_i=f"{m['acc_i']:.3f}",
                        loss=f"{loss.item():.2f}", refresh=False)

        if step % args.log_every == 0:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            a = {k: run[k] / seen for k in run}
            bar.write(f"step {step:>5}/{args.steps}  "
                      f"acc_t {a['acc_t']:.4f}  acc_i {a['acc_i']:.4f}  "
                      f"ce_t {a['ce_t']:.3f}  ce_i {a['ce_i']:.3f}  "
                      f"cyc {a['cycle']:.3f}  imgrec {a['img_rec']:.3f}  kl {a['kl']:.2f}  "
                      f"vram {mem:.2f}GB")
            run = {}; seen = 0
        if step % args.sample_every == 0:
            dump_samples(model, sample_tokens, step, args.sample_dir, cfg, device)
            bar.write(f"  -> wrote sample ti_step{step:06d}.png/.txt")
        if step % args.save_every == 0:
            save_ckpt(step)
        if step >= args.steps:
            break
    bar.close()

    save_ckpt(step)
    at, ai = dump_samples(model, sample_tokens, step, args.sample_dir, cfg, device)
    print(f"saved {args.ckpt}  |  final sample acc_t={at:.3f} acc_i={ai:.3f}  "
          f"-> {args.sample_dir}/ti_step{step:06d}.png")


if __name__ == "__main__":
    main()
