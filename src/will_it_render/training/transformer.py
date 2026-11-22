from __future__ import annotations

import torch
import torch.nn as nn


class TransformerProbe(nn.Module):
    """Transformer probe over the (layer, head) axes of a stacked attention feature.

    Adapts the ``CrossAttentionExtractor`` from ``learn-to-steer`` (Sapir et al.,
    github.com/sapir1779/learn-to-steer, ``train_relation_classifier/model.py``) — which
    tokenizes an *image* diffusion model's per-layer cross-attention maps and aggregates
    them with a transformer + a learnable query — to our *video* features, and to a binary
    "does the target class appear in the final clip" objective.

    Two adaptations vs the reference:

    1. **Heads are an explicit token axis** (the reference pre-averaged heads).
    2. **Factorized attention** to keep every sequence short (≤ max(H, L)) despite the
       ``L*H`` = 1536-token product:
         - STAGE 1 — a shared learnable query pools the ``H`` head tokens *within* each
           layer → one token per layer.
         - STAGE 2 — a pre-norm ``TransformerEncoder`` self-attends across the ``L`` layer
           tokens, then a second learnable query pools them → one vector → MLP head → logit.

    Input contract (identical to the ``resnet_layer_agg`` "stack" path): the feature is the
    captured blocks stacked block-major, ``[B, L*H, T, h, w]`` (or flat ``[B, D]``), where
    ``L = n_blocks`` and ``H = n_blocks_heads // n_blocks``. The temporal axis ``T`` — which
    the reference image model does not have — is folded into each (layer, head) token's
    feature by the per-token tokenizer (a Linear over the flattened ``T*h*w`` map).

    Alternatives (not the default): a *flat* transformer over all ``L*H`` tokens jointly
    (more expressive but ~1 GB+ attention scores/layer — rejected for memory), or a temporal
    mean-pool then a per-layer Linear over the ``h*w`` map (closest to the reference image
    model, but discards temporal structure).
    """

    def __init__(
        self,
        n_blocks_heads: int,               # L*H (e.g. 1536)
        latent_thw: tuple[int, int, int],  # (T, h, w) of the stored latent
        n_blocks: int,                     # L — splits n_blocks_heads into [L, H]
        d_model: int = 256,
        nhead: int = 4,                    # stage-2 layer-transformer heads
        num_layers: int = 3,               # stage-2 transformer depth
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        head_attn_nheads: int = 4,         # stage-1 head-pooling attention heads
        layer_attn_nheads: int = 4,        # stage-2 query-pooling attention heads
        token_hidden: int = 0,             # 0 → single Linear tokenizer; >0 → 2-layer MLP
        token_mode: str = "linear",        # "linear" (flatten→Linear) | "conv" (2D conv over h,w, T as channels)
        token_conv_ch: int = 32,           # conv-tokenizer channel width
        mlp_hidden: int = 128,             # classifier head hidden width
        pool: str = "query",               # "query" (learned query attn) | "mean" (baseline)
        timestep_mode: str = "none",       # "none" | "pre" | "mid" (off by default; per-step probes fix the step)
        num_timesteps: int = 0,            # only used when timestep_mode != "none"
    ) -> None:
        super().__init__()
        if n_blocks <= 0 or n_blocks_heads % n_blocks != 0:
            raise ValueError(
                f"TransformerProbe requires n_blocks > 0 dividing n_blocks_heads "
                f"(got n_blocks={n_blocks}, n_blocks_heads={n_blocks_heads})."
            )
        for name, nh in (("nhead", nhead), ("head_attn_nheads", head_attn_nheads),
                         ("layer_attn_nheads", layer_attn_nheads)):
            if d_model % nh != 0:
                raise ValueError(f"d_model={d_model} must be divisible by {name}={nh}.")
        if pool not in ("query", "mean"):
            raise ValueError(f"pool must be 'query' | 'mean' (got {pool!r}).")
        if timestep_mode not in ("none", "pre", "mid"):
            raise ValueError(f"timestep_mode must be 'none' | 'pre' | 'mid' (got {timestep_mode!r}).")

        self.n_blocks_heads = n_blocks_heads
        self.latent_thw = latent_thw
        self.n_blocks = n_blocks
        self.n_heads = n_blocks_heads // n_blocks
        self.d_model = d_model
        self.pool = pool
        self.timestep_mode = timestep_mode
        self.token_mode = token_mode

        T, h, w = latent_thw
        token_in = T * h * w

        # (a) Per-(layer, head) tokenizer — each head's [T, h, w] map → one d_model token.
        # "linear": flatten the map → Linear (faithful to the reference's Linear(map²→d),
        #   extended to the spatiotemporal map). Cheap, but discards spatial structure.
        # "conv": a small 2D conv over (h, w) with T as input channels → pooled → Linear,
        #   preserving the maps' spatial structure before projecting to the token.
        if token_mode == "conv":
            self.token_conv = nn.Sequential(
                nn.Conv2d(T, token_conv_ch, 3, padding=1), nn.SiLU(inplace=True),
                nn.Conv2d(token_conv_ch, token_conv_ch, 3, stride=2, padding=1), nn.SiLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.tokenizer: nn.Module = nn.Linear(token_conv_ch, d_model)
        elif token_hidden > 0:
            self.tokenizer = nn.Sequential(
                nn.Linear(token_in, token_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(token_hidden, d_model),
            )
        else:
            self.tokenizer = nn.Linear(token_in, d_model)

        # Learned identity embeddings over the head and layer axes (2-D learned pos-enc over
        # the L×H grid), so the attention layers know which layer/head each token is.
        self.head_embed = nn.Embedding(self.n_heads, d_model)
        self.layer_embed = nn.Embedding(self.n_blocks, d_model)

        # (b) STAGE 1 — learnable query pools the H head tokens within each layer (shared
        # across layers; applied to the folded [B*L, H, d] sequence).
        self.head_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.head_attn = nn.MultiheadAttention(d_model, head_attn_nheads, dropout=dropout,
                                               batch_first=True)

        # (d, optional) Denoising-step embedding (reference's timestep embedding). Off by
        # default — per-step probes fix the step, so this is only meaningful for a future
        # multi-step pooled variant.
        if timestep_mode != "none":
            if num_timesteps <= 0:
                raise ValueError("timestep_mode != 'none' requires num_timesteps > 0.")
            self.step_embed: nn.Embedding | None = nn.Embedding(num_timesteps, d_model)
        else:
            self.step_embed = None

        # (c) STAGE 2 — pre-norm transformer over the L layer tokens.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # (d) Pool over the L layer tokens with a second learnable query.
        if pool == "query":
            self.layer_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.layer_attn = nn.MultiheadAttention(d_model, layer_attn_nheads, dropout=dropout,
                                                    batch_first=True)

        # (e) Classifier head → binary logit.
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, mlp_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None,
                return_pooled: bool = False) -> torch.Tensor:
        """``cond`` is the per-sample denoising-step index ``[B]`` (long) for a
        step-conditioned variant; consumed only when ``timestep_mode != "none"`` and
        ignored otherwise, so single-step callers use ``model(x)`` unchanged.
        ``return_pooled=True`` returns the pooled ``[B, d_model]`` vector *before* the
        classifier head — the CVG-style compact per-step representation, for a multi-step
        aggregator that combines these across denoising steps."""
        T, h, w = self.latent_thw
        B = x.shape[0]
        L, H, d = self.n_blocks, self.n_heads, self.d_model

        # [B, D] or [B, C, T, h, w] → [B, L, H, T, h, w]
        x = x.reshape(B, self.n_blocks_heads, T, h, w)

        # (a) tokenize each (layer, head) map → [B, L, H, d]
        if self.token_mode == "conv":
            c = self.token_conv(x.reshape(B * L * H, T, h, w)).flatten(1)  # [B*L*H, conv_ch]
            tok = self.tokenizer(c).reshape(B, L, H, d)
        else:
            tok = self.tokenizer(x.reshape(B, L, H, T * h * w))
        tok = tok + self.head_embed.weight[None, None, :, :]  # add head identity over H

        # (b) STAGE 1 — pool H head tokens within each layer (fold L into batch).
        kv = tok.reshape(B * L, H, d)
        q = self.head_query.expand(B * L, 1, d)
        layer_tok, _ = self.head_attn(q, kv, kv)          # [B*L, 1, d]
        layer_tok = layer_tok.reshape(B, L, d)

        # (c) STAGE 2 — transformer over the L layer tokens.
        layer_tok = layer_tok + self.layer_embed.weight[None, :, :]  # add layer identity over L
        step_emb = None
        if self.step_embed is not None and cond is not None:
            step_emb = self.step_embed(cond.long().view(-1)).unsqueeze(1)  # [B, 1, d]
        if self.timestep_mode == "pre" and step_emb is not None:
            layer_tok = layer_tok + step_emb
        if self.timestep_mode == "mid" and step_emb is not None:
            for layer in self.transformer.layers:
                layer_tok = layer(layer_tok + step_emb)
        else:
            layer_tok = self.transformer(layer_tok)       # [B, L, d]

        # (d) pool over layers.
        if self.pool == "query":
            ql = self.layer_query.expand(B, 1, d)
            pooled, _ = self.layer_attn(ql, layer_tok, layer_tok)  # [B, 1, d]
            pooled = pooled.squeeze(1)
        else:
            pooled = layer_tok.mean(dim=1)                # [B, d]

        if return_pooled:
            return pooled  # [B, d] — CVG-style compact per-step representation (pre-head)
        # (e) head → logit [B, 1]
        return self.head(pooled)
