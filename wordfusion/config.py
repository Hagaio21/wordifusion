"""Central configuration for WORDFUSION.

Everything that controls model/data size lives here so we can scale the whole
pipeline up toward the GPU's limit by editing one place.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- data / image geometry ---
    img_h: int = 16          # grid height  (chars per column)
    img_w: int = 16          # grid width   (chars per row)
    corpus_path: str = "data/tinyshakespeare.txt"

    # --- VAE ---
    latent_ch: int = 8       # channels in the latent
    vae_base_ch: int = 96    # base width of the VAE conv stack
    downsample: int = 2      # spatial downsample factor per side (16 -> 8)

    # --- text<->image VAE (bidirectional, learned codec; no palette) ---
    # text -> z (ti_latent_ch, hw, hw) -> image (3, image_size, image_size) -> z -> text
    image_size: int = 64        # z' side: RGB image_size x image_size
    text_max_len: int = 1024    # max sequence length (tokens/positions) the text heads handle
    text_min_len: int = 8       # min content length sampled during training
    corpus_dir: str = "data"    # dir of .txt corpora for the char-level VAE
    ti_random_frac: float = 0.5  # fraction of training windows that are random ASCII chars
                                  # (de-overfit: don't memorize one corpus's statistics)
    ti_latent_ch: int = 4       # channels in the shared latent z
    ti_downsample: int = 4      # image_size -> latent side (64 -> 16); power of 2
    ti_base_ch: int = 64        # conv width of the image encoder/decoder
    ti_text_dim: int = 256      # transformer width of the text encoder/decoder
    ti_text_layers: int = 4     # transformer depth (each of the two text heads)
    ti_text_heads: int = 4      # attention heads
    # robustness: force the text to survive a REAL image round-trip
    ti_quantize: bool = True    # round the image to 8-bit pixels before re-encoding
    ti_img_noise: float = 0.0   # gaussian noise added to the image before re-encoding
    # loss weights
    ti_beta_kl: float = 1e-4    # KL weight (small: diffusion handles sampleability)
    ti_lambda_cycle: float = 1.0  # image faithfully stores z  (encode(image) ~ z)
    ti_lambda_img: float = 1.0    # image round-trip: decode(encode(image)) ~ image

    # --- diffusion UNet ---
    unet_base_ch: int = 128
    unet_ch_mult: tuple = (1, 2, 2)
    timesteps: int = 1000

    # --- training ---
    batch_size: int = 128
    lr: float = 2e-4
    device: str = "cuda"
    amp: bool = False        # fp32: plenty of VRAM, and fp16 caused VAE NaNs
    seed: int = 0

    @property
    def capacity(self) -> int:
        return self.img_h * self.img_w

    @property
    def latent_hw(self) -> tuple:
        return (self.img_h // self.downsample, self.img_w // self.downsample)

    @property
    def ti_latent_hw(self) -> int:
        return self.image_size // self.ti_downsample   # 64 // 4 = 16


DEFAULT = Config()
