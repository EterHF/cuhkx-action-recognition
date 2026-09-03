"""R(2+1)D-34 architecture used by the released visual branch."""

from torch import nn
from torchvision.models.video.resnet import (
    BasicBlock,
    Conv2Plus1D,
    R2Plus1dStem,
    VideoResNet,
)


def r2plus1d_34(num_classes: int = 400) -> nn.Module:
    model = VideoResNet(
        block=BasicBlock,
        conv_makers=[Conv2Plus1D] * 4,
        layers=[3, 4, 6, 3],
        stem=R2Plus1dStem,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Match the Caffe2/VMZ layout used by the source checkpoint.
    model.layer2[0].conv2[0] = Conv2Plus1D(128, 128, 288)
    model.layer3[0].conv2[0] = Conv2Plus1D(256, 256, 576)
    model.layer4[0].conv2[0] = Conv2Plus1D(512, 512, 1152)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm3d):
            module.eps = 1e-3
            module.momentum = 0.9
    return model
