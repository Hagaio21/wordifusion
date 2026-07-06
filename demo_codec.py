"""Round-trip demo for the text<->image codec.

- Encodes a sample string to a colored image.
- Saves both the raw (H,W) image and an upscaled viewable PNG.
- Decodes it back and checks it matches.
- Adds gaussian noise and shows decode is robust (this simulates VAE/diffusion error).
"""

import numpy as np
from PIL import Image

from wordfusion import encode_text, decode_image, upscale, PALETTE, NUM_TOKENS

H, W = 16, 16

sample = "Hello, WORDFUSION!\nturning text into colored pixels :)"

img = encode_text(sample, H, W)
recovered = decode_image(img)

print(f"vocab size: {NUM_TOKENS}, palette shape: {PALETTE.shape}")
print(f"image shape: {img.shape}  (capacity = {H*W} chars)")
print(f"input : {sample!r}")
print(f"decoded: {recovered!r}")
print(f"lossless round-trip: {recovered == sample}")

# robustness: add noise like a VAE/diffusion model would introduce
for sigma in (10, 20, 40, 60):
    noisy = img.astype(np.float32) + np.random.normal(0, sigma, img.shape)
    dec = decode_image(noisy)
    ok = dec == sample
    print(f"  sigma={sigma:>3}: exact_recover={ok}  ({sum(a==b for a,b in zip(dec,sample))}/{len(sample)} chars)")

Image.fromarray(img).save("sample_raw.png")
Image.fromarray(upscale(img, 16)).save("sample_view.png")

# also save the full palette as a strip for reference
strip = upscale(PALETTE.reshape(1, -1, 3), 24)
Image.fromarray(strip).save("palette.png")
print("saved: sample_raw.png, sample_view.png, palette.png")
