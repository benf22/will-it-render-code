"""``build_attnsteer`` — instantiate a probe architecture by name.

Hoisted out of ``training.trainer``. The steering path
needs only this factory, but importing it from ``training.trainer`` dragged in the whole
training stack (augment, evaluation, summarize_run, probe_dataset, hydra).
This module imports **only** the leaf probe-architecture classes, so ``import
common.probe_factory`` is cheap. ``training.trainer`` re-exports ``build_attnsteer`` for
backwards compatibility.
"""
from __future__ import annotations

from typing import Optional

import torch.nn as nn

from will_it_render.training.attn_pool import AttnPoolProbe
from will_it_render.training.conv import ConvProbe
from will_it_render.training.maxpool import MaxPoolProbe
from will_it_render.training.mlp import ProbeMLP
from will_it_render.training.resnet import ResNetProbe
from will_it_render.training.transformer import TransformerProbe


def build_attnsteer(
    probe_model: str,
    n_blocks_heads: int,
    latent_thw: tuple[int, int, int],
    bottleneck_ch: int = 128,
    conv_kernel_size: int = 3,
    mlp_hidden: int = 128,
    dropout: float = 0.3,
    mlp_hidden_dims: tuple[int, ...] = (512, 128),
    n_blocks: int = 0,
    n_heads: int = 0,
    attn_pool_conv_ch: int = 32,
    attn_pool_mlp_hidden: int = 64,
    resnet_base_ch: int = 16,
    resnet_n_blocks: int = 3,
    resnet_groups: int = 8,
    resnet_temporal_spatial: bool = True,
    resnet_head_dropout: float = 0.0,
    resnet_dropblock_size: int = 0,
    resnet_norm: str = "group",
    resnet_layer_agg: str = "none",
    resnet_layer_agg_k: int = 1,
    resnet_layer_agg_hidden: int = 16,
    resnet_input_norm: str = "none",
    resnet_layer_agg_init_blocks: Optional[list[int]] = None,
    resnet_layer_agg_init_random: bool = False,
    resnet_step_cond: bool = False,
    resnet_step_embed_dim: int = 64,
    resnet_perframe: bool = False,
    resnet_perframe_head: str = "shared",
    resnet_perframe_tap: str = "post_temporal",
    resnet_classhead: bool = False,
    resnet_n_classes: int = 0,
    resnet_class_head_arch: str = "pooled",
    resnet_class_mlp_hidden: int = 128,
    transformer_d_model: int = 256,
    transformer_nhead: int = 4,
    transformer_num_layers: int = 3,
    transformer_dim_feedforward: int = 512,
    transformer_head_attn_nheads: int = 4,
    transformer_layer_attn_nheads: int = 4,
    transformer_token_hidden: int = 0,
    transformer_token_mode: str = "linear",
    transformer_token_conv_ch: int = 32,
    transformer_mlp_hidden: int = 128,
    transformer_pool: str = "query",
    transformer_timestep_mode: str = "none",
    transformer_num_timesteps: int = 0,
) -> nn.Module:
    """Instantiate a probe model by name."""
    if probe_model == "conv":
        return ConvProbe(
            n_blocks_heads=n_blocks_heads,
            latent_thw=latent_thw,
            bottleneck_ch=bottleneck_ch,
            conv_kernel_size=conv_kernel_size,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    if probe_model == "mlp":
        T, h, w = latent_thw
        return ProbeMLP(
            input_dim=n_blocks_heads * T * h * w,
            hidden_dims=mlp_hidden_dims,
            dropout=dropout,
        )
    if probe_model == "attn_pool":
        if n_blocks == 0 or n_heads == 0:
            raise ValueError("attn_pool requires n_blocks and n_heads > 0")
        return AttnPoolProbe(
            n_blocks=n_blocks,
            n_heads=n_heads,
            latent_thw=latent_thw,
            conv_ch=attn_pool_conv_ch,
            mlp_hidden=attn_pool_mlp_hidden,
            dropout=dropout,
        )
    if probe_model == "maxpool":
        return MaxPoolProbe(
            n_blocks_heads=n_blocks_heads,
            latent_thw=latent_thw,
            mlp_hidden=mlp_hidden,
            dropout=dropout,
        )
    if probe_model == "resnet":
        return ResNetProbe(
            n_blocks_heads=n_blocks_heads,
            latent_thw=latent_thw,
            base_ch=resnet_base_ch,
            n_res_blocks=resnet_n_blocks,
            groups=resnet_groups,
            dropout=dropout,
            temporal_spatial=resnet_temporal_spatial,
            head_dropout=resnet_head_dropout,
            dropblock_size=resnet_dropblock_size,
            norm=resnet_norm,
            layer_agg=resnet_layer_agg,
            n_blocks=n_blocks,
            layer_agg_k=resnet_layer_agg_k,
            layer_agg_hidden=resnet_layer_agg_hidden,
            layer_agg_init_blocks=resnet_layer_agg_init_blocks,
            layer_agg_init_random=resnet_layer_agg_init_random,
            step_cond=resnet_step_cond,
            step_embed_dim=resnet_step_embed_dim,
            perframe=resnet_perframe,
            perframe_head=resnet_perframe_head,
            perframe_tap=resnet_perframe_tap,
            input_norm=resnet_input_norm,
            classhead=resnet_classhead,
            n_classes=resnet_n_classes,
            class_head_arch=resnet_class_head_arch,
            class_mlp_hidden=resnet_class_mlp_hidden,
        )
    if probe_model == "transformer":
        return TransformerProbe(
            n_blocks_heads=n_blocks_heads,
            latent_thw=latent_thw,
            n_blocks=n_blocks,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            head_attn_nheads=transformer_head_attn_nheads,
            layer_attn_nheads=transformer_layer_attn_nheads,
            token_hidden=transformer_token_hidden,
            token_mode=transformer_token_mode,
            token_conv_ch=transformer_token_conv_ch,
            mlp_hidden=transformer_mlp_hidden,
            pool=transformer_pool,
            timestep_mode=transformer_timestep_mode,
            num_timesteps=transformer_num_timesteps,
        )
    raise ValueError(
        f"Unknown probe_model {probe_model!r}. Choose 'conv', 'mlp', 'attn_pool', "
        f"'maxpool', 'resnet', or 'transformer'."
    )
