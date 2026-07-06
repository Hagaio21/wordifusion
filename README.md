# WORDFUSION

A *weird LLM*: instead of predicting text left-to-right one token at a time, WORDFUSION
generates a whole passage at once with **diffusion** — in the latent space of images
where **each character is a colored pixel**.

```
text ──codec──▶ colored image ──VAE encode──▶ latent ──▶ [ diffusion model ] ──▶ latent
                                                                                    │
text ◀──text decoder── colored image ◀──VAE decode──────────────────────────────────┘
        (nearest color)
```

## The pipeline

1. **Codec** (`wordfusion/codec.py`) — text ⇄ colored image.
   Each of 97 characters (printable ASCII + newline + PAD) maps to a distinct RGB color
   from a coarse 5×5×5 grid. A string is laid out row-major, one char per pixel.
   Decoding an image is a **separate stage**: nearest-palette-color lookup per pixel.
   The wide color spacing is what lets decoding survive VAE + diffusion noise.

2. **VAE** (`wordfusion/vae.py`) — a pure *image* autoencoder.
   Encoder: RGB `(3,H,W)` → latent `(latent_ch, H/d, W/d)`. Decoder: latent → RGB.
   It knows nothing about text; turning its output back into characters is the codec's job.

3. **Latent diffusion** (`wordfusion/diffusion.py`) — the generative model ("the LLM").
   A small time-conditioned UNet + DDPM (cosine schedule, ε-prediction) that denoises the
   VAE latent. Models `p(latent)` over encoded text passages.
   **Prompting = inpainting**: freeze the prompt's latent cells, diffuse the rest.

## Usage (in the `tq` conda env, which has CUDA torch)

```bash
python scripts/prep_data.py                    # download corpus + smoke-test codec/data
python scripts/train_vae.py --steps 8000       # train the image VAE
python scripts/train_diffusion.py --steps 8000 # train latent diffusion on the frozen VAE
python scripts/generate.py --n 8               # sample text from noise
python scripts/generate.py --prompt "To be, "  # prompt via inpainting
```

Everything scalable (image size, latent size, model width/depth) lives in
`wordfusion/config.py`. Start small, then scale toward the GPU's limit.

## Status

Working end-to-end vertical slice at 16×16 (256 chars/passage), latent 4×8×8.
Corpus: TinyShakespeare. Next: push image/latent/model size up to fill VRAM.
