"""FlowTransformerRefiner — DiT-style transformer that predicts a rectified-flow
velocity over a patchified TSDF target (default 64^3).

Conditioning enters in three places:
  1. **Coarse voxel patches**: the Qwen voxel64 is kept at its native 64^3,
     patchified the same way as the target, and concatenated to each target
     token along the channel dim (input to the first projection).
  2. **Visual feature**: small CNN encoder -> [B, F=256] vector. Added to the
     timestep embedding to form the AdaLN modulation signal.
  3. **Timestep**: sinusoidal embedding -> MLP -> shared modulation input.

Both (2) and (3) drive AdaLN-Zero modulation in every transformer block. The
"-Zero" trick (scale and gate initialized to 0) lets the network start as an
identity mapping w.r.t. the residual stream, which empirically stabilizes
flow / diffusion training (Peebles & Xie 2023, DiT).

Architecture defaults (small): hidden=384, depth=6, heads=6.
Token grid at defaults: (64/4)^3 = 16^3 = 4096 tokens, each carrying 64 dims
(input proj sees 128 = 2 * 4^3 from target+coarse concat). Attention is the
dominant compute since it scales O(N^2) in sequence length.
Param budget: ~19.6M trainable params with default settings.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from simfishlib.experimental.flow_transformer_refiner.target import PatchTokenizer
from simfishlib.experimental.flow_transformer_refiner.visual_encoder import VisualEncoder


# ---------------------------------------------------------------------------
# Timestep & modulation
# ---------------------------------------------------------------------------


def sinusoidal_time_embedding(t: torch.Tensor, dim: int, max_period: float = 1000.0) -> torch.Tensor:
    """t: [B] in [0, 1] -> [B, dim] sinusoidal features."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=t.dtype)
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :] * max_period
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLNZero(nn.Module):
    """Predicts (shift, scale, gate) from a conditioning vector and applies them
    to a token stream.

    The output `gate` multiplies the *residual* branch in DiT blocks (i.e. the
    output of attention / MLP before adding back to the stream). Initialized to
    zero so the block is an identity at init.
    """

    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(cond_dim, 3 * hidden_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s, c, g = self.proj(F.silu(cond)).chunk(3, dim=-1)
        return s.unsqueeze(1), c.unsqueeze(1), g.unsqueeze(1)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


# ---------------------------------------------------------------------------
# Transformer block (DiT-style)
# ---------------------------------------------------------------------------


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, cond_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.mod_attn = AdaLNZero(hidden_dim, cond_dim)
        self.mod_mlp = AdaLNZero(hidden_dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, c1, g1 = self.mod_attn(cond)
        h = modulate(self.norm1(x), s1, c1)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1 * attn_out

        s2, c2, g2 = self.mod_mlp(cond)
        h = modulate(self.norm2(x), s2, c2)
        x = x + g2 * self.mlp(h)
        return x


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------


class FlowTransformerRefiner(nn.Module):
    def __init__(
        self,
        resolution: int = 64,
        patch: int = 4,
        target_channels: int = 1,
        coarse_channels: int = 1,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int = 384,
        visual_feature_dim: int = 256,
        use_visual: bool = True,
    ) -> None:
        super().__init__()
        self.tokenizer = PatchTokenizer(resolution=resolution, patch=patch)
        self.resolution = resolution
        self.patch = patch
        self.target_channels = target_channels
        self.coarse_channels = coarse_channels
        self.hidden_dim = hidden_dim
        self.use_visual = use_visual
        self.visual_feature_dim = visual_feature_dim

        patch_dim = self.tokenizer.patch_volume
        target_tok_dim = target_channels * patch_dim
        coarse_tok_dim = coarse_channels * patch_dim
        self.input_proj = nn.Linear(target_tok_dim + coarse_tok_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.tokenizer.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # conditioning vector: timestep + (optional) visual feature
        if use_visual:
            self.visual_encoder = VisualEncoder(feature_dim=visual_feature_dim)
            self.visual_proj = nn.Linear(visual_feature_dim, cond_dim)
        else:
            self.visual_encoder = None
            self.visual_proj = None
        self.time_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        self.cond_dim = cond_dim
        self.time_embed_dim = cond_dim

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim) for _ in range(depth)]
        )

        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_mod = AdaLNZero(hidden_dim, cond_dim)
        # final projection initialized to zero so the model starts predicting v=0
        self.output_proj = nn.Linear(hidden_dim, target_tok_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    # -- helpers ------------------------------------------------------------

    def encode_condition(
        self,
        t: torch.Tensor,
        visual: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build the [B, cond_dim] conditioning vector consumed by AdaLN."""
        t_emb = sinusoidal_time_embedding(t.to(dtype), self.time_embed_dim)
        c = self.time_mlp(t_emb)
        if self.use_visual:
            if visual is None:
                vis_feat = torch.zeros(batch_size, self.visual_feature_dim, device=device, dtype=dtype)
            else:
                vis_feat = self.visual_encoder(visual)
            c = c + self.visual_proj(vis_feat)
        return c

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        x_t: torch.Tensor,                 # [B, target_channels, R, R, R]
        coarse: torch.Tensor,              # [B, coarse_channels, R, R, R]
        t: torch.Tensor,                   # [B] in [0,1]
        visual: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity v_theta(x_t, t, coarse, visual) at the same shape as x_t."""
        if x_t.shape != coarse.shape and x_t.shape[2:] != coarse.shape[2:]:
            raise ValueError(
                f"x_t {tuple(x_t.shape)} and coarse {tuple(coarse.shape)} must match spatially"
            )
        B = x_t.shape[0]
        device, dtype = x_t.device, x_t.dtype

        x_tokens = self.tokenizer.patchify(x_t)
        c_tokens = self.tokenizer.patchify(coarse)
        tokens = self.input_proj(torch.cat([x_tokens, c_tokens], dim=-1))
        tokens = tokens + self.pos_embed

        cond = self.encode_condition(t, visual, B, device, dtype)
        for block in self.blocks:
            tokens = block(tokens, cond)

        s, c, _ = self.final_mod(cond)
        tokens = modulate(self.final_norm(tokens), s, c)
        out_tokens = self.output_proj(tokens)
        return self.tokenizer.unpatchify(out_tokens, channels=self.target_channels)


def build_model(
    resolution: int = 64,
    patch: int = 4,
    hidden_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    visual_feature_dim: int = 256,
    use_visual: bool = True,
    cond_dim: int | None = None,
) -> FlowTransformerRefiner:
    cond_dim = cond_dim if cond_dim is not None else hidden_dim
    return FlowTransformerRefiner(
        resolution=resolution,
        patch=patch,
        hidden_dim=hidden_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_dim=cond_dim,
        visual_feature_dim=visual_feature_dim,
        use_visual=use_visual,
    )
