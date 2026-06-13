import torch
import torch.nn as nn
import torch.nn.functional as F

from network.models import init_weights


class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=8, ngf=64, norm_layer=nn.InstanceNorm2d):
        super(UNetGenerator, self).__init__()

        # Construct U-Net structure
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

        init_weights(self)

    def forward(self, x):
        return self.model(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super(ResidualConvBlock, self).__init__()
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, bias=False),
            norm_layer(out_channels),
        )
        self.activation = nn.ReLU(True)

    def forward(self, x):
        return self.activation(self.block(x) + self.shortcut(x))


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super(DownsampleBlock, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2, True),
            ResidualConvBlock(out_channels, out_channels, norm_layer),
        )

    def forward(self, x):
        return self.model(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, norm_layer=nn.InstanceNorm2d):
        super(UpsampleBlock, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
        )
        self.fuse = ResidualConvBlock(out_channels + skip_channels, out_channels, norm_layer)

    def forward(self, x, skip):
        x = self.up(x)
        return self.fuse(torch.cat([x, skip], dim=1))


class ResUNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=16, norm_layer=nn.InstanceNorm2d):
        super(ResUNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.enc1 = ResidualConvBlock(input_nc, ngf, norm_layer)
        self.enc2 = DownsampleBlock(ngf, ngf * 2, norm_layer)
        self.enc3 = DownsampleBlock(ngf * 2, ngf * 4, norm_layer)
        self.enc4 = DownsampleBlock(ngf * 4, ngf * 8, norm_layer)
        self.enc5 = DownsampleBlock(ngf * 8, ngf * 8, norm_layer)

        self.bottleneck = nn.Sequential(
            ResidualConvBlock(ngf * 8, ngf * 8, norm_layer),
            ResidualConvBlock(ngf * 8, ngf * 8, norm_layer),
        )

        self.dec4 = UpsampleBlock(ngf * 8, ngf * 8, ngf * 8, norm_layer)
        self.dec3 = UpsampleBlock(ngf * 8, ngf * 4, ngf * 4, norm_layer)
        self.dec2 = UpsampleBlock(ngf * 4, ngf * 2, ngf * 2, norm_layer)
        self.dec1 = UpsampleBlock(ngf * 2, ngf, ngf, norm_layer)

        self.out = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ngf, output_nc, kernel_size=3),
        )

        init_weights(self)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        z = self.bottleneck(e5)
        d4 = self.dec4(z, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        residual = self.out(d1)
        if self.input_nc == self.output_nc:
            return torch.tanh(x + residual)
        return torch.tanh(residual)


class UNetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.InstanceNorm2d):
        super(UNetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(PatchGANDiscriminator, self).__init__()

        kw = 4  # Kernel size
        padw = 1  # Padding
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Output 1 channel prediction per patch
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

        init_weights(self)

    def forward(self, x):
        x = self.model(x)
        return x


class SpectralNormDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, use_dropout=False):
        super(SpectralNormDiscriminator, self).__init__()

        model = [
            nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, True),

            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1)),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.utils.spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1)),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, True),
        ]

        if use_dropout:
            model.append(nn.Dropout2d(0.5))

        model += [
            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
        ]

        self.model = nn.Sequential(*model)

        init_weights(self)

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    # Create a U-Net generator instance with 'xavier' initialization
    # net = UNetGenerator(input_nc=1, output_nc=1).to('cuda')
    net = SpectralNormDiscriminator(input_nc=1).to('cuda')
    # Print initialized weights
    # for name, param in net.named_parameters():
    #     print(name, param.data)

    from torchsummary import summary

    summary(net, input_size=(1, 256, 256))
