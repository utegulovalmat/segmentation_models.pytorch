"""
history:
v1:
CBR filters: (conv64-bn-relu) (conv128-bn-relu) (conv128-1) sigmoid

v2:
CBR filters: (conv32-bn-relu) (conv64-bn-relu) (conv128-bn-relu)
(conv256-bn-relu) (conv512-bn-relu) (conv512x1) sigmoid
"""
import torch
import torch.nn as nn


class Conv2dReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        super().__init__()
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.ReLU(inplace=True)

        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ConvNet(nn.Module):
    def __init__(
        self, size: str = "small64", in_channels: int = 1, classes: int = 1,
    ):
        super(ConvNet, self).__init__()
        """
        v2:
        === 5 down - 5 up
        CBR-Large32-512 filters: (conv32-bn-relu) maxpool (conv64-bn-relu) maxpool
        (conv128-bn-relu) maxpool (conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
        ... sigmoid

        === 4 down - 4 up
        CBR-Large64-512 filters: (conv64-bn-relu) maxpool (conv128-bn-relu) maxpool
        (conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
        ... sigmoid

        CBR-Small32-256 filters: (conv32-bn-relu) maxpool (conv64-bn-relu) maxpool
        (conv128-bn-relu) maxpool (conv256-bn-relu) maxpool
        ... sigmoid

        CBR-Small64-512 filters: (conv64-bn-relu) maxpool (conv128-bn-relu) maxpool
        (conv256-bn-relu) maxpool (conv512-bn-relu) maxpool
        ... sigmoid
        """
        self.size = size
        # 32 for input in_channels
        self.conv32 = Conv2dReLU(
            in_channels, 32, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv32 = Conv2dReLU(
            32, classes, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # 64 for input in_channels
        self.conv64_input = Conv2dReLU(
            in_channels, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv64_output = Conv2dReLU(
            64, classes, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )

        # 64
        self.conv64 = Conv2dReLU(
            32, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv64 = Conv2dReLU(
            64, 32, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # 128
        self.conv128 = Conv2dReLU(
            64, 128, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv128 = Conv2dReLU(
            128, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # 256
        self.conv256 = Conv2dReLU(
            128, 256, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv256 = Conv2dReLU(
            256, 128, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # 512
        self.conv512 = Conv2dReLU(
            256, 512, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.deconv512 = Conv2dReLU(
            512, 256, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # other
        self.last256 = nn.Conv2d(256, classes, kernel_size=1, stride=1)
        self.last512 = nn.Conv2d(512, classes, kernel_size=1, stride=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.size in ["small32", "large32"]:
            x = self.conv32(x)
            x = self.maxpool(x)  # 256
            x = self.conv64(x)
            x = self.maxpool(x)  # 128
        else:
            x = self.conv64_input(x)
            x = self.maxpool(x)  # 256

        x = self.conv128(x)
        x = self.maxpool(x)  # 128 / 64
        x = self.conv256(x)
        x = self.maxpool(x)  # 64 / 32

        # bottleneck
        if self.size in ["small64", "large32", "large64"]:
            x = self.conv512(x)

        x = self.deconv256(x)
        x = self.up(x)  # 128 / 64
        x = self.deconv128(x)
        x = self.up(x)  # 256 / 128

        if self.size in ["small32", "large32"]:
            x = self.deconv64(x)
            x = self.up(x)  # 256
            x = self.deconv32(x)
            x = self.up(x)  # 512
        else:
            x = self.deconv64_output(x)
            x = self.up(x)  # 512

        out = self.sigmoid(x)
        return out

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        self.eval()
        with torch.no_grad():
            x = self.forward(x)
        return x
