from unet import *


class UNetSmall(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNetSmall, self).__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)

        # Bottleneck
        self.bottleneck1 = DownSample(64, 128)

        # Decoder
        self.dec4 = UpSample(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)

        # Bottleneck
        b = self.bottleneck1(e1)

        # Decoder
        d4 = self.dec4(b, e1)

        return self.out_conv(d4)


if __name__ == "__main__":
    rand_x = torch.randn((1, 3, 512, 512))
    model = UNetSmall(in_channels=3, out_channels=1)
    output = model(rand_x)
    print(output.shape)
