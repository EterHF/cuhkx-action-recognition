"""Minimal DSTFormer implementation compatible with the released checkpoint."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(inputs)))))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, temporal: bool, drop: float) -> None:
        super().__init__()
        self.num_heads = heads
        self.scale = (dim // heads) ** -0.5
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.mode = "temporal" if temporal else "spatial"
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, inputs: torch.Tensor, frames: int) -> torch.Tensor:
        batch_frames, joints, channels = inputs.shape
        qkv = (
            self.qkv(inputs)
            .reshape(
                batch_frames, joints, 3, self.num_heads, channels // self.num_heads
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        if self.mode == "spatial":
            attention = (query @ key.transpose(-2, -1)) * self.scale
            output = self.attn_drop(attention.softmax(dim=-1)) @ value
            output = output.transpose(1, 2).reshape(batch_frames, joints, channels)
        else:
            query = query.reshape(
                -1, frames, self.num_heads, joints, channels // self.num_heads
            ).permute(0, 2, 3, 1, 4)
            key = key.reshape(
                -1, frames, self.num_heads, joints, channels // self.num_heads
            ).permute(0, 2, 3, 1, 4)
            value = value.reshape(
                -1, frames, self.num_heads, joints, channels // self.num_heads
            ).permute(0, 2, 3, 1, 4)
            attention = (query @ key.transpose(-2, -1)) * self.scale
            output = self.attn_drop(attention.softmax(dim=-1)) @ value
            output = output.permute(0, 3, 2, 1, 4).reshape(
                batch_frames, joints, channels
            )
        return self.proj_drop(self.proj(output))


class Block(nn.Module):
    def __init__(
        self, dim: int, heads: int, mlp_ratio: float, drop: float, mode: str
    ) -> None:
        super().__init__()
        self.st_mode = mode
        self.norm1_s = nn.LayerNorm(dim)
        self.norm1_t = nn.LayerNorm(dim)
        self.attn_s = Attention(dim, heads, temporal=False, drop=drop)
        self.attn_t = Attention(dim, heads, temporal=True, drop=drop)
        self.drop_path = nn.Identity()
        self.norm2_s = nn.LayerNorm(dim)
        self.norm2_t = nn.LayerNorm(dim)
        self.mlp_s = MLP(dim, int(dim * mlp_ratio), drop)
        self.mlp_t = MLP(dim, int(dim * mlp_ratio), drop)
        self.att_fuse = False

    def forward(self, inputs: torch.Tensor, frames: int) -> torch.Tensor:
        if self.st_mode == "stage_st":
            inputs = inputs + self.attn_s(self.norm1_s(inputs), frames)
            inputs = inputs + self.mlp_s(self.norm2_s(inputs))
            inputs = inputs + self.attn_t(self.norm1_t(inputs), frames)
            return inputs + self.mlp_t(self.norm2_t(inputs))
        inputs = inputs + self.attn_t(self.norm1_t(inputs), frames)
        inputs = inputs + self.mlp_t(self.norm2_t(inputs))
        inputs = inputs + self.attn_s(self.norm1_s(inputs), frames)
        return inputs + self.mlp_s(self.norm2_s(inputs))


class DSTformer(nn.Module):
    def __init__(
        self,
        dim_in: int = 3,
        dim_out: int = 3,
        dim_feat: int = 256,
        dim_rep: int = 512,
        depth: int = 5,
        num_heads: int = 8,
        mlp_ratio: float = 4,
        num_joints: int = 17,
        maxlen: int = 243,
        drop_rate: float = 0.0,
        att_fuse: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(drop_rate)
        self.blocks_st = nn.ModuleList(
            [
                Block(dim_feat, num_heads, mlp_ratio, drop_rate, "stage_st")
                for _ in range(depth)
            ]
        )
        self.blocks_ts = nn.ModuleList(
            [
                Block(dim_feat, num_heads, mlp_ratio, drop_rate, "stage_ts")
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim_feat)
        self.pre_logits = (
            nn.Sequential(
                OrderedDict(
                    (
                        ("fc", nn.Linear(dim_feat, dim_rep)),
                        ("act", nn.Tanh()),
                    )
                )
            )
            if dim_rep
            else nn.Identity()
        )
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity()
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        self.att_fuse = att_fuse
        if att_fuse:
            self.ts_attn = nn.ModuleList(
                [nn.Linear(dim_feat * 2, 2) for _ in range(depth)]
            )

    def forward(self, inputs: torch.Tensor, return_rep: bool = False) -> torch.Tensor:
        batch, frames, joints, channels = inputs.shape
        value = self.joints_embed(inputs.reshape(-1, joints, channels)) + self.pos_embed
        value = value.reshape(batch, frames, joints, -1) + self.temp_embed[:, :frames]
        value = self.pos_drop(value.reshape(batch * frames, joints, -1))
        for index, (spatial_first, temporal_first) in enumerate(
            zip(self.blocks_st, self.blocks_ts)
        ):
            spatial = spatial_first(value, frames)
            temporal = temporal_first(value, frames)
            if self.att_fuse:
                alpha = self.ts_attn[index](
                    torch.cat((spatial, temporal), dim=-1)
                ).softmax(dim=-1)
                value = spatial * alpha[:, :, :1] + temporal * alpha[:, :, 1:]
            else:
                value = (spatial + temporal) * 0.5
        value = self.pre_logits(self.norm(value).reshape(batch, frames, joints, -1))
        return value if return_rep else self.head(value)

    def get_representation(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.forward(inputs, return_rep=True)
