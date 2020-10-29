from torch import nn
from torch.nn import functional as F


class LinearBlock(nn.Sequential):
    def __init__(self, in_d, out_d):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_d, out_d),
            nn.BatchNorm1d(out_d),
        )


class Conv2dBlock(nn.Sequential):
    def __init__(self, in_c, out_c):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
        )


class Upsample2d(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor,
                             mode='bilinear')


class Generator(nn.Sequential):
    def __init__(self, in_d, body_c):
        c = body_c
        super().__init__(
            nn.Linear(in_d, in_d * 4),
            nn.BatchNorm1d(in_d * 4),
            LinearBlock(in_d * 4, c * 16),
            Reshape(c, 4, 4),  # 4.
            Conv2dBlock(c, c, 3, 1, 1),
            Upsample2d(2),  # 8.
            Conv2dBlock(c, c, 3, 1, 1),
            Upsample2d(2),  # 16.
            Conv2dBlock(c, c, 3, 1, 1),
            Upsample2d(2),  # 32.
            Conv2dBlock(c, c, 3, 1, 1),
            Upsample2d(2),  # 64.
            Conv2dBlock(c, c, 3, 1, 1),
            nn.Conv2d(c, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
