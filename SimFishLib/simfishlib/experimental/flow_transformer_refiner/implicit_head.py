"""Implicit-decoder variant of the flow-transformer refiner.

The dense head in ``model.py`` outputs a (1, R, R, R) TSDF volume; the supervision
resolution and the inference resolution are stuck together. This module replaces
that head with a Convolutional-Occupancy-Networks / DeepSDF-style pair:

  1. The DiT transformer (same as before, but conditioned only on coarse) emits a
     small **feature volume** of shape [B, C, G, G, G], where G = R / patch.
  2. A tiny MLP decoder takes any continuous query point ``(x, y, z)`` in
     [0,1]^3, trilinearly samples the feature volume, concatenates Fourier
     positional encoding, the current flow state ``x_t`` at that point, and the
     timestep ``t``, then predicts the rectified-flow velocity at that point.

Inference can then march cubes at *any* resolution by querying the MLP on the
desired grid — there is no longer a 64^3 ceiling on output detail.

Flow-matching parameterization is unchanged. Only the head differs.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

from simfishlib.experimental.flow_transformer_refiner.model import (
    AdaLNZero,
    DiTBlock,
    modulate,
    sinusoidal_time_embedding,
)
from simfishlib.experimental.flow_transformer_refiner.target import PatchTokenizer
from simfishlib.experimental.flow_transformer_refiner.visual_encoder import VisualEncoder


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def _fourier_positional_encoding(xyz: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """xyz: [..., 3] in [0, 1] -> [..., 3 + 6*num_freqs] Fourier features."""
    if num_freqs <= 0:
        return xyz
    freqs = torch.pow(
        2.0,
        torch.arange(num_freqs, device=xyz.device, dtype=xyz.dtype),
    ) * math.pi  # [num_freqs]
    args = xyz.unsqueeze(-1) * freqs  # [..., 3, num_freqs]
    sin = torch.sin(args).flatten(-2)
    cos = torch.cos(args).flatten(-2)
    return torch.cat([xyz, sin, cos], dim=-1)


def _trilinear_sample_feature_volume(
    feature_volume: torch.Tensor,  # [B, C, G, G, G] with axes (X, Y, Z)
    query_xyz: torch.Tensor,        # [B, K, 3] in [0, 1]
) -> torch.Tensor:
    """Trilinearly sample a 3D feature grid at scattered query points.

    F.grid_sample expects volumes laid out as [B, C, D, H, W] with the grid's
    last dim ordered as (x, y, z) mapping to (W, H, D). We store the volume as
    (X, Y, Z); permuting to (Z, Y, X) makes the grid's (x, y, z) -> (W, H, D)
    match our (X, Y, Z) convention.

    Returns [B, K, C].
    """
    B, K = query_xyz.shape[:2]
    vol = feature_volume.permute(0, 1, 4, 3, 2).contiguous()  # [B, C, Z, Y, X]
    grid = (query_xyz * 2.0 - 1.0).view(B, K, 1, 1, 3)        # last dim (x, y, z)
    samp = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)
    # samp: [B, C, K, 1, 1]
    return samp.view(B, samp.shape[1], K).permute(0, 2, 1).contiguous()


class ImplicitSDFDecoder(nn.Module):
    """MLP that reads (sampled feature, Fourier pos, x_t-at-q, t) -> velocity.

    The final linear is zero-initialized so the model predicts velocity ~= 0 at
    init, mirroring the AdaLN-Zero stable-start trick used by the encoder.
    """

    def __init__(
        self,
        feature_dim: int,
        num_freqs: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")
        self.num_freqs = num_freqs
        pos_dim = 3 + 3 * 2 * num_freqs if num_freqs > 0 else 3
        # input = feature + pos + x_t (1) + t (1)
        in_dim = feature_dim + pos_dim + 1 + 1
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.out = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.feature_dim = feature_dim

    def forward(
        self,
        feature_volume: torch.Tensor,  # [B, C, G, G, G]
        query_xyz: torch.Tensor,        # [B, K, 3]
        x_t_at_q: torch.Tensor,         # [B, K, 1] or [B, K]
        t: torch.Tensor,                # [B]
    ) -> torch.Tensor:
        """Returns predicted velocity at the query points, shape [B, K, 1]."""
        B, K = query_xyz.shape[:2]
        feat = _trilinear_sample_feature_volume(feature_volume, query_xyz)  # [B, K, C]
        pos = _fourier_positional_encoding(query_xyz, self.num_freqs)        # [B, K, pos_dim]
        if x_t_at_q.dim() == 2:
            x_t_at_q = x_t_at_q.unsqueeze(-1)
        t_in = t.view(B, 1, 1).expand(B, K, 1)
        inp = torch.cat([feat, pos, x_t_at_q, t_in], dim=-1)
        h = self.trunk(inp)
        return self.out(h)


# ---------------------------------------------------------------------------
# Encoder + head wrapper
# ---------------------------------------------------------------------------


class FlowTransformerRefinerImplicit(nn.Module):
    """DiT encoder over the coarse voxel -> feature volume -> implicit decoder.

    Key differences from the dense ``FlowTransformerRefiner``:
      - The transformer can take either just the coarse condition, or the
        coarse condition concatenated with a 64^3 grid view of the current
        flow state ``x_t`` (``dit_xt_channels > 0``). The latter gives the
        encoder voxel-level state about *where along the trajectory* we are,
        which the original implicit head lacked.
      - The output of the transformer is a feature volume, not a TSDF volume.
      - ``forward`` takes per-point (x_t_at_q, query_xyz) and returns per-point
        velocity. When ``dit_xt_channels > 0``, ``forward`` also requires the
        grid-level ``x_t_grid`` so the encoder can be conditioned on it.
    """

    def __init__(
        self,
        coarse_resolution: int = 64,
        patch: int = 4,
        coarse_channels: int = 1,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        cond_dim: int | None = None,
        visual_feature_dim: int = 256,
        use_visual: bool = True,
        feature_dim: int = 128,
        decoder_hidden: int = 256,
        decoder_layers: int = 5,
        decoder_num_freqs: int = 10,
        dit_xt_channels: int = 0,
    ) -> None:
        super().__init__()
        cond_dim = cond_dim if cond_dim is not None else hidden_dim
        self.coarse_tokenizer = PatchTokenizer(resolution=coarse_resolution, patch=patch)
        self.coarse_resolution = coarse_resolution
        self.patch = patch
        self.coarse_channels = coarse_channels
        self.dit_xt_channels = int(dit_xt_channels)
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.G = self.coarse_tokenizer.grid
        self.use_visual = use_visual
        self.visual_feature_dim = visual_feature_dim
        self.cond_dim = cond_dim
        self.time_embed_dim = cond_dim

        patch_dim = self.coarse_tokenizer.patch_volume
        in_channels = coarse_channels + self.dit_xt_channels
        self.input_proj = nn.Linear(in_channels * patch_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.coarse_tokenizer.num_tokens, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

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

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_dim, num_heads, mlp_ratio, cond_dim) for _ in range(depth)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.final_mod = AdaLNZero(hidden_dim, cond_dim)
        self.feature_proj = nn.Linear(hidden_dim, feature_dim)

        self.decoder = ImplicitSDFDecoder(
            feature_dim=feature_dim,
            num_freqs=decoder_num_freqs,
            hidden_dim=decoder_hidden,
            num_layers=decoder_layers,
        )

    # -- conditioning --------------------------------------------------------

    def encode_condition(
        self,
        t: torch.Tensor,
        visual: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t.to(dtype), self.time_embed_dim)
        c = self.time_mlp(t_emb)
        if self.use_visual:
            if visual is None:
                vis_feat = torch.zeros(batch_size, self.visual_feature_dim, device=device, dtype=dtype)
            else:
                vis_feat = self.visual_encoder(visual)
            c = c + self.visual_proj(vis_feat)
        return c

    # -- transformer encoder -> feature volume -------------------------------

    def encode(
        self,
        coarse: torch.Tensor,                 # [B, coarse_channels, R, R, R]
        t: torch.Tensor,                       # [B]
        visual: torch.Tensor | None = None,
        x_t_grid: torch.Tensor | None = None,  # [B, dit_xt_channels, R, R, R]
    ) -> torch.Tensor:
        """Run the transformer over the coarse condition and return a feature volume.

        When ``self.dit_xt_channels > 0``, ``x_t_grid`` is concatenated to ``coarse``
        on the channel axis before patchifying, so the encoder sees the current
        grid-level flow state alongside the static condition.

        Output shape: [B, feature_dim, G, G, G] with axes interpreted as (X, Y, Z).
        """
        B = coarse.shape[0]
        device, dtype = coarse.device, coarse.dtype
        if self.dit_xt_channels > 0:
            if x_t_grid is None:
                raise ValueError(
                    f"dit_xt_channels={self.dit_xt_channels}: encode() requires x_t_grid"
                )
            if x_t_grid.shape[2:] != coarse.shape[2:]:
                raise ValueError(
                    f"x_t_grid spatial {tuple(x_t_grid.shape[2:])} != "
                    f"coarse spatial {tuple(coarse.shape[2:])}"
                )
            stacked = torch.cat([coarse, x_t_grid.to(device=device, dtype=dtype)], dim=1)
            c_tokens = self.coarse_tokenizer.patchify(stacked)
        else:
            c_tokens = self.coarse_tokenizer.patchify(coarse)
        tokens = self.input_proj(c_tokens) + self.pos_embed

        cond = self.encode_condition(t, visual, B, device, dtype)
        for block in self.blocks:
            tokens = block(tokens, cond)

        s, c, _ = self.final_mod(cond)
        tokens = modulate(self.final_norm(tokens), s, c)
        feat_tokens = self.feature_proj(tokens)  # [B, N, feature_dim]

        # Tokens were emitted in (X, Y, Z) traversal order by PatchTokenizer
        # (it splits each axis into grid*patch and permutes 0, 2, 4, 6, ...).
        # Reshape back to [B, G, G, G, feature_dim] then to [B, C, G, G, G].
        G = self.G
        feat_volume = feat_tokens.view(B, G, G, G, self.feature_dim).permute(0, 4, 1, 2, 3).contiguous()
        return feat_volume

    # -- forward (flow-matching velocity at query points) --------------------

    def forward(
        self,
        x_t_at_q: torch.Tensor,        # [B, K, 1] or [B, K]
        coarse: torch.Tensor,           # [B, 1, R, R, R]
        t: torch.Tensor,                # [B]
        query_xyz: torch.Tensor,        # [B, K, 3] in [0, 1]
        visual: torch.Tensor | None = None,
        x_t_grid: torch.Tensor | None = None,  # [B, dit_xt_channels, R, R, R]
    ) -> torch.Tensor:
        feature_volume = self.encode(coarse, t, visual, x_t_grid=x_t_grid)
        return self.decoder(feature_volume, query_xyz, x_t_at_q, t)


def build_implicit_model(
    coarse_resolution: int = 64,
    patch: int = 4,
    hidden_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
    cond_dim: int | None = None,
    visual_feature_dim: int = 256,
    use_visual: bool = True,
    feature_dim: int = 128,
    decoder_hidden: int = 256,
    decoder_layers: int = 5,
    decoder_num_freqs: int = 10,
    dit_xt_channels: int = 0,
) -> FlowTransformerRefinerImplicit:
    return FlowTransformerRefinerImplicit(
        coarse_resolution=coarse_resolution,
        patch=patch,
        hidden_dim=hidden_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        cond_dim=cond_dim,
        visual_feature_dim=visual_feature_dim,
        use_visual=use_visual,
        feature_dim=feature_dim,
        decoder_hidden=decoder_hidden,
        decoder_layers=decoder_layers,
        decoder_num_freqs=decoder_num_freqs,
        dit_xt_channels=dit_xt_channels,
    )
