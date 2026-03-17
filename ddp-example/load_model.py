import torch
import torch.nn as nn
from typing import List, Type


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


class ResNetCIFAR(nn.Module):
    """
    CIFAR-style ResNet.
    Differences from ImageNet ResNet:
    - 3x3 stem
    - no initial maxpool
    - suited for 32x32 inputs
    """

    def __init__(
        self,
        block: Type[BasicBlock],
        num_blocks: List[int],
        num_classes: int = 10,
        base_width: int = 64,
    ):
        super().__init__()
        self.in_planes = base_width

        self.conv1 = nn.Conv2d(
            3,
            base_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, base_width,   num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18_cifar(num_classes: int = 10, base_width: int = 64) -> nn.Module:
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        base_width=base_width,
    )


def resnet34_cifar(num_classes: int = 10, base_width: int = 64) -> nn.Module:
    return ResNetCIFAR(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        num_classes=num_classes,
        base_width=base_width,
    )


def build_model(model_name: str, num_classes: int = 10, base_width: int = 64) -> nn.Module:
    if model_name == "resnet18":
        return resnet18_cifar(num_classes=num_classes, base_width=base_width)
    if model_name == "resnet34":
        return resnet34_cifar(num_classes=num_classes, base_width=base_width)
    raise ValueError(f"Unsupported model_name={model_name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())