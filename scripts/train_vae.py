"""Train the WORDFUSION VAE (text-image autoencoder with a char-logit decoder)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from wordfusion.config import DEFAULT as cfg
from wordfusion.data import ShakespeareWindows, render_rgb, nearest_char_idx
from wordfusion.vae import VAE, vae_loss
from wordfusion.codec import idx_to_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=cfg.batch_size)
    ap.add_argument("--lr", type=float, default=cfg.lr)
    ap.add_argument("--beta", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default="checkpoints/vae.pt")
    args = ap.parse_args()

    device = cfg.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)

    ds = ShakespeareWindows(cfg.corpus_path, cfg.img_h, cfg.img_w,
                            epoch_size=args.steps * args.batch, seed=cfg.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=(device == "cuda"))

    model = VAE(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device == "cuda")

    print(f"device={device}  params={n_params/1e6:.2f}M  latent={cfg.latent_ch}x{cfg.latent_hw}  "
          f"img={cfg.img_h}x{cfg.img_w}  batch={args.batch}")

    model.train()
    t0 = time.time()
    run_acc = run_recon = run_kl = 0.0
    seen = 0
    for step, idx in enumerate(dl, 1):
        idx = idx.to(device, non_blocking=True)
        rgb = render_rgb(idx, device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=cfg.amp and device == "cuda"):
            recon_img, mean, logvar = model(rgb)
            loss, recon, kl = vae_loss(recon_img, rgb, mean, logvar, beta=args.beta)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                   # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        with torch.no_grad():
            # text-decode the reconstruction and compare to ground-truth chars
            acc = (nearest_char_idx(recon_img.float()) == idx).float().mean().item()
        run_acc += acc; run_recon += recon.item(); run_kl += kl.item(); seen += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            print(f"step {step:>5}/{args.steps}  acc {run_acc/seen:.4f}  "
                  f"recon {run_recon/seen:.4f}  kl {run_kl/seen:.3f}  "
                  f"{step*args.batch/dt:.0f} img/s  vram {mem:.2f}GB")
            run_acc = run_recon = run_kl = 0.0; seen = 0
        if step % 500 == 0:
            torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, args.ckpt)
        if step >= args.steps:
            break

    torch.save({"model": model.state_dict(), "cfg": vars(cfg)}, args.ckpt)
    print(f"saved {args.ckpt}")

    # qualitative check: reconstruct a few windows
    model.eval()
    idx = next(iter(DataLoader(ds, batch_size=4, num_workers=0)))
    with torch.no_grad():
        recon_img, _, _ = model(render_rgb(idx, device))
        pred = nearest_char_idx(recon_img.float()).cpu()
    for i in range(4):
        orig = idx_to_text(idx[i].numpy())[:80]
        rec = idx_to_text(pred[i].numpy())[:80]
        match = (pred[i] == idx[i]).float().mean().item()
        print(f"\n[{i}] acc={match:.3f}\n  orig: {orig!r}\n  rec : {rec!r}")


if __name__ == "__main__":
    main()
