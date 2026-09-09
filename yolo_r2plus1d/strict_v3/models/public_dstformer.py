"""Released CUHK-X DSTFormer classification head."""

from __future__ import annotations

import torch
from torch import nn

from yolo_r2plus1d.strict_v3.models.dstformer import DSTformer


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = DSTformer(
            dim_in=3,
            dim_out=3,
            dim_feat=64,
            dim_rep=64,
            depth=1,
            num_heads=2,
            mlp_ratio=2,
            maxlen=1,
            num_joints=17,
            att_fuse=True,
        )
        self.head = nn.ModuleDict(
            {
                "drop": nn.Dropout(0.5),
                "bn": nn.BatchNorm1d(2048),
                "fc1": nn.Linear(64 * 17, 2048),
                "relu": nn.ReLU(inplace=True),
                "fc2": nn.Linear(2048, 40),
            }
        )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the released 2048-D representation immediately before fc2."""
        features = self.backbone.get_representation(inputs)
        features = features.reshape(inputs.shape[0], inputs.shape[1], 17, -1)
        features = features.permute(0, 2, 3, 1).mean(-1).reshape(inputs.shape[0], -1)
        features = self.head["drop"](features)
        features = self.head["fc1"](features)
        features = self.head["bn"](features)
        return self.head["relu"](features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head["fc2"](self.forward_features(inputs))
