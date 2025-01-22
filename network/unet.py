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
    def __init__(self, input_nc, ndf=64, n_layers=3):
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
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            nn.InstanceNorm2d(ndf * nf_mult),
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
    def __init__(self, input_nc, ndf=64):
        super(SpectralNormDiscriminator, self).__init__()

        self.model = nn.Sequential(
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

            nn.utils.spectral_norm(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1))
        )

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
