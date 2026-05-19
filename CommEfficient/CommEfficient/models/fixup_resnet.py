# import torch
# import torch.nn as nn
# import numpy as np
# # from fixup.imagenet.models.fixup_resnet_imagenet import FixupResNet, FixupBottleneck


# from .resnets import ResNet as FixupResNet
# from .resnets import Bottleneck as FixupBottleneck


# __all__ = ["FixupResNet50"]

# class FixupResNet50(FixupResNet):
#     def __init__(self, **kwargs):
#         super().__init__(FixupBottleneck, [3, 4, 6, 3], **kwargs)




import torch
import torch.nn as nn
import numpy as np
# from fixup.imagenet.models.fixup_resnet_imagenet import FixupResNet, FixupBottleneck

from .resnets import ResNet as FixupResNet
from .resnets import Bottleneck as FixupBottleneck

__all__ = ["FixupResNet50", "FixupBasicBlock", "conv3x3"]

# ---- CIFAR-style Fixup BasicBlock ----
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class FixupBasicBlock(nn.Module):
    """CIFAR-style Fixup BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(FixupBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)

        # Fixup initialization trick: learnable scale and biases
        self.scale = nn.Parameter(torch.ones(1))
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.bias2b = nn.Parameter(torch.zeros(1))

        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Conv2d(inplanes, planes,
                                      kernel_size=1, stride=stride, bias=True)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = x + self.bias1a
        out = self.conv1(out)
        out = out + self.bias1b
        out = self.relu(out)

        out = out + self.bias2a
        out = self.conv2(out)
        out = out * self.scale + self.bias2b

        out += self.shortcut(x)
        return self.relu(out)

# ---- ImageNet FixupResNet50 ----
class FixupResNet50(FixupResNet):
    def __init__(self, **kwargs):
        super().__init__(FixupBottleneck, [3, 4, 6, 3], **kwargs)
