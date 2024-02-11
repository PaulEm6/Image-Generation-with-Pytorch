# discriminator_generator_module.py
import torch.nn as nn

class DiscriminatorGeneratorModule:
    @staticmethod
    def create_discriminator(nc, ndf):
        discriminator = nn.Sequential(

        # input is ``(nc) x 64 x 64``
        nn.Conv2d(in_channels=nc, out_channels=ndf, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. ``(ndf) x 32 x 32``
        nn.Conv2d(in_channels=ndf, out_channels=ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. ``(ndf*2) x 16 x 16``
        nn.Conv2d(in_channels=ndf * 2, out_channels=ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. ``(ndf*4) x 8 x 8``
        nn.Conv2d(in_channels=ndf * 4, out_channels=ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        # state size. ``(ndf*8) x 4 x 4``
        nn.Conv2d(in_channels=ndf * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
        nn.Sigmoid())

        # Initialize weights with a normal distribution of mean 0 and std 0.02
        for m in discriminator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        return discriminator

    @staticmethod
    def create_generator(nz, ngf, nc):
        generator = nn.Sequential(

        # input is Z, going into a convolution
        nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),

        # state size. ``(ngf*8) x 4 x 4``
        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),

        # state size. ``(ngf*4) x 8 x 8``
        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        # state size. ``(ngf*2) x 16 x 16``
        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        # state size. ``(ngf) x 32 x 32``
        nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
        nn.Tanh()
        # state size. ``(nc) x 64 x 64``
        )

        # state size. ``(nc) x 64 x 64``

        # Initialize weights with a normal distribution of mean 0 and std 0.02
        for m in generator.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.02)

        return generator