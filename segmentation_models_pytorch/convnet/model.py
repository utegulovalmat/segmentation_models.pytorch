import torch
import torch.nn as nn
from torchvision import models


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
        self,
        # encoder_name: str = "resnet34",
        encoder_depth: int = 2,
        # encoder_weights: str = "imagenet",
        # decoder_use_batchnorm: bool = True,
        # decoder_channels: List[int] = (256, 128, 64, 32, 16),
        # decoder_attention_type: Optional[str] = None,
        in_channels: int = 1,
        classes: int = 1,
        # activation: Optional[Union[str, callable]] = None,
        # aux_params: Optional[dict] = None,
    ):
        super(ConvNet, self).__init__()
        self.conv1 = Conv2dReLU(
            in_channels, 64, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        self.conv2 = Conv2dReLU(
            64, 128, kernel_size=3, stride=1, padding=1, use_batchnorm=True
        )
        # TODO: encoder_depth more
        self.conv_last = nn.Conv2d(128, classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv1(x)
        # up1 = self.upsample1(x)
        # x = self.layer2(x)
        # up2 = self.upsample2(x)
        # x = self.layer3(x)
        # up3 = self.upsample3(x)
        # x = self.layer4(x)
        # up4 = self.upsample4(x)
        # merge = torch.cat([up1, up2, up3, up4], dim=1)
        # merge = self.conv1k(merge)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv_last(x)
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
