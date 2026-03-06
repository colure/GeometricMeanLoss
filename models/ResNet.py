"""
ResNet backbone adapted in part from torchvision's implementation:
https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["resnet10", "resnet12", "resnet18", "resnet34", "resnet50"]


def _conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def _conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = x + residual
        return self.relu(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        shortcut: nn.Module | None = None,
    ) -> None:
        super().__init__()
        bottleneck_channels = out_channels
        expanded_channels = out_channels * self.expansion

        self.conv1 = _conv1x1(in_channels, bottleneck_channels)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = _conv3x3(bottleneck_channels, bottleneck_channels, stride)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = _conv1x1(bottleneck_channels, expanded_channels)
        self.bn3 = nn.BatchNorm2d(expanded_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = shortcut

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x = x + residual
        return self.relu(x)


class ResNet(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        layers: Sequence[int],
        widths: Sequence[int] | None = None,
        feature_dim: int = 512,
        num_classes: int = 64,
        projection: bool = False,
        zero_init_residual: bool = False,
        drop_rate: float = 0,
        use_fc: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if drop_rate > 0:
            raise NotImplementedError("Dropout is not currently supported")
        if widths is None:
            widths = [64, 128, 256, 512]
        if len(widths) != 4:
            raise ValueError("widths must contain four stage widths")

        self.use_fc = use_fc
        self.projection = projection
        self.current_channels = widths[0]

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(block, widths[0], layers[0], stride=1)
        self.stage2 = self._make_stage(block, widths[1], layers[1], stride=2)
        self.stage3 = self._make_stage(block, widths[2], layers[2], stride=2)
        self.stage4 = self._make_stage(block, widths[3], layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        embedding_dim = widths[3] * block.expansion
        if projection:
            self.projection_head = nn.Linear(embedding_dim, feature_dim)
            self.proj1 = self.projection_head
        if use_fc:
            self.classifier = nn.Linear(embedding_dim, num_classes)
            self.fc = self.classifier

        self._initialize_weights(zero_init_residual)

    def _make_stage(
        self,
        block: type[BasicBlock] | type[Bottleneck],
        out_channels: int,
        depth: int,
        stride: int,
    ) -> nn.Sequential:
        shortcut = self._make_shortcut(out_channels, block.expansion, stride)
        blocks = [block(self.current_channels, out_channels, stride=stride, shortcut=shortcut)]
        self.current_channels = out_channels * block.expansion

        for _ in range(1, depth):
            blocks.append(block(self.current_channels, out_channels))

        return nn.Sequential(*blocks)

    def _make_shortcut(
        self,
        out_channels: int,
        expansion: int,
        stride: int,
    ) -> nn.Module | None:
        target_channels = out_channels * expansion
        if stride == 1 and self.current_channels == target_channels:
            return None
        return nn.Sequential(
            _conv1x1(self.current_channels, target_channels, stride),
            nn.BatchNorm2d(target_channels),
        )

    def _initialize_weights(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        if not zero_init_residual:
            return

        for module in self.modules():
            if isinstance(module, Bottleneck):
                nn.init.constant_(module.bn3.weight, 0)
            elif isinstance(module, BasicBlock):
                nn.init.constant_(module.bn2.weight, 0)

    def _flatten_pooled(self, x: Tensor) -> Tensor:
        return torch.flatten(self.pool(x), 1)

    def _forward_backbone(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.stem(x)
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        return stage1, stage2, stage3, stage4

    def forward(
        self,
        x: Tensor,
        use_fc: bool = False,
        cat: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        stage1, stage2, stage3, stage4 = self._forward_backbone(x)

        if cat and not self.training:
            features = torch.cat(
                [
                    self._flatten_pooled(stage2),
                    self._flatten_pooled(stage3),
                    self._flatten_pooled(stage4),
                ],
                dim=1,
            )
            logits = None
        else:
            features = self._flatten_pooled(stage4)
            logits = self.classifier(features) if use_fc and self.use_fc else None

        if self.training and self.projection:
            features = self.projection_head(features)

        return features, logits


def _build_resnet(
    block: type[BasicBlock] | type[Bottleneck],
    layers: Sequence[int],
    **kwargs,
) -> ResNet:
    return ResNet(block, layers, **kwargs)


def resnet10(**kwargs) -> ResNet:
    return _build_resnet(BasicBlock, [1, 1, 1, 1], **kwargs)


def resnet12(**kwargs) -> ResNet:
    return _build_resnet(BasicBlock, [1, 1, 2, 1], widths=[64, 160, 320, 640], **kwargs)


def resnet18(**kwargs) -> ResNet:
    return _build_resnet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs) -> ResNet:
    return _build_resnet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return _build_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
