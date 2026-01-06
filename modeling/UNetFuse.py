import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, N):
        super().__init__()
        self.double_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.double_conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.features_max = nn.Linear(N, N)
        self.features_avg = nn.Linear(N, N)
        #self.w = nn.Parameter(torch.zeros(N, N))


        

    def forward(self, x):
        B,C,L,N = x.shape
        x = self.double_conv1(x)
        residual_x = x
        avg_x = self.features_avg(F.adaptive_avg_pool2d(x, (1, N)))  # B,C,1,N
        max_x = self.features_max(F.adaptive_max_pool2d(x, (1, N)))  # B,C,1,N
        #w = self.w * F.softmax(max_x-avg_x)  # B,C,N,N x B,C,1,N 调整离散程度
        #x = torch.einsum("bctn, bcmn->bctm", x, F.softmax(w))  # B,C,L,N
        x = x * F.softmax(max_x-avg_x)
        x = (residual_x + x)/2.0
        x = self.double_conv2(x)
        return x


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, N):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 1)),
            DoubleConv(in_channels, out_channels, N)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, N, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=(2, 1), stride=(2, 1))

        self.conv = DoubleConv(in_channels, out_channels, N)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetFuse(nn.Module):
    def __init__(self, n_channels, out_channels, N, bilinear=True):
        super(UNetFuse, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(self.n_channels, 64, N)
        self.down1 = Down(64, 128, N)
        self.down2 = Down(128, 256, N)
        self.down3 = Down(256, 512, N)
        self.down4 = Down(512, 512, N)
        self.up1 = Up(1024, 256, N, bilinear)
        self.up2 = Up(512, 128, N, bilinear)
        self.up3 = Up(256, 64, N, bilinear)
        self.up4 = Up(128, 64, N, bilinear)
        self.outc = OutConv(64, out_channels)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)

        return out
