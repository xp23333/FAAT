import torch
import torch.nn as nn


class PixelDiscriminator(nn.Module):
    def __init__(self, nc=640, ndf=512, n_classes=10):
        super(PixelDiscriminator, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(ndf, ndf // 2, 4, 2, 1, bias=False)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.cls1 = nn.Conv2d(ndf // 2, self.n_classes, 2, 1, 0, bias=False)
        self.cls2 = nn.Conv2d(ndf // 2, self.n_classes, 2, 1, 0, bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.lrelu1(out)

        out = self.conv2(out)
        out = self.lrelu2(out)

        src_out = self.cls1(out)
        tar_out = self.cls2(out)

        return torch.cat((src_out, tar_out), dim=1).view(-1, self.n_classes * 2)
