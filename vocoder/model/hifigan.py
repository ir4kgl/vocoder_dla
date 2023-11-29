import torch
import torch.nn as nn
from torch.nn.utils import weight_norm, spectral_norm


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def forward(self, x):
        return nn.functional.leaky_relu(x, 0.1)


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlock, self).__init__()
        self.layers = nn.ModuleList()
        for d in dilations:
            self.layers.append(nn.Sequential(
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                        dilation=d, padding=get_padding(kernel_size, d))),
                LeakyReLU(),
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                        dilation=1, padding=get_padding(kernel_size, 1))),
                LeakyReLU()
            ))
        for seq in self.layers:
            for l in seq:
                if "Conv1d" in l.__class__.__name__:
                    nn.init.normal_(l.weight)
                    nn.init.constant_(l.bias, 0.0)

    def forward(self, x):
        for layer in self.layers:
            x_residual = layer(x)
            x = x + x_residual
        return x


class MRF(nn.Module):
    def __init__(self, c, kernel_sizes, dilations):
        super(MRF, self).__init__()
        self.resblocks = nn.ModuleList()
        for i in range(len(kernel_sizes)):
            self.resblocks.append(ResBlock(c, kernel_sizes[i], dilations[i]))

    def forward(self, x):
        y = self.resblocks[0](x)        
        for layer in self.resblocks[1:]:
            y = y + layer(x)
        return y / len(self.resblocks)


class Upsampler(nn.Module):
    def __init__(self, c, upsample_rate, kernel_size):
        super(Upsampler, self).__init__()
        self.t_conv = weight_norm(nn.ConvTranspose1d(
            c, c // 2, kernel_size, stride=upsample_rate,
            padding=(kernel_size - upsample_rate) // 2
        ))
        nn.init.normal_(self.t_conv.weight)
        nn.init.constant_(self.t_conv.bias, 0.0)

    def forward(self, x):
        return self.t_conv(x)


class GeneratorBlock(nn.Module):
    def __init__(self, upsample_rate, upsample_kernel_size,
                 c, mrf_kernel_sizes, mrf_dilations):
        super(GeneratorBlock, self).__init__()
        self.activation = LeakyReLU()
        self.upsampler = Upsampler(c, upsample_rate, upsample_kernel_size)
        self.mrf = MRF(c // 2, mrf_kernel_sizes, mrf_dilations)

    def forward(self, x):
        x = self.activation(x)
        x = self.upsampler(x)
        x = self.mrf(x)
        return x


class Generator(nn.Module):
    def __init__(self, upsample_rates, upsample_kernel_sizes,
                            mrf_kernel_sizes, mrf_dilations, c=512):
        super(Generator, self).__init__()
        assert(len(upsample_rates) == len(upsample_kernel_sizes))
        self.n_blocks = len(upsample_rates)
        self.preconv = weight_norm(nn.Conv1d(80, c, 7, 1, padding=3))
        self.blocks = []
        for i in range(self.n_blocks):
            self.blocks.append(GeneratorBlock(
                upsample_rates[i], upsample_kernel_sizes[i], c,
                mrf_kernel_sizes, mrf_dilations))
            c //= 2
        self.blocks = nn.Sequential(*self.blocks)
        self.activation = nn.LeakyReLU()
        self.postconv = weight_norm(nn.Conv1d(c, 1, 7, 1, padding=3))
        nn.init.normal_(self.preconv.weight)
        nn.init.constant_(self.preconv.bias, 0.0)
        nn.init.normal_(self.postconv.weight)
        nn.init.constant_(self.postconv.bias, 0.0)

    def forward(self, x):
        x = self.preconv(x)
        x = self.blocks(x)
        x = self.activation(x)
        x = self.postconv(x)
        return torch.tanh(x)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super(PeriodDiscriminator, self).__init__()
        self.period = period
        f_pad = (get_padding(5, 1), 0)
        self.layers = nn.ModuleList()
        in_c = 1
        for c in [32, 128, 512, 1024]:
            self.layers.append(
                weight_norm(nn.Conv2d(in_c, c, (kernel_size, 1), (stride, 1), f_pad)))
            in_c = c
        self.layers.append(weight_norm(nn.Conv2d(in_c, in_c, (kernel_size, 1), padding=(2,0))))
        self.activation = LeakyReLU()
        self.postconv = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def pad(self, x):
        t = x.shape[-1]
        if t % self.period != 0:
            x = torch.nn.functional.pad(
                x, (0, self.period - (t % self.period)), "reflect")
        return x

    def forward(self, x):
        feature_map = []
        x = self.pad(x)
        x = x.view(x.shape[0], x.shape[1], x.shape[-1] // self.period, self.period)
        for l in self.layers:
            x = l(x)
            x = self.activation(x)
            feature_map.append(x)
        x = self.postconv(x)
        feature_map.append(x)
        return torch.flatten(x, 1, -1), feature_map


class ScaleDiscriminator(nn.Module):
    def __init__(self, norm):
        super(ScaleDiscriminator, self).__init__()
        self.layers = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, 7)),
            norm(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.activation = LeakyReLU()
        self.postconv = norm(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x):
        feature_map = []
        for l in self.layers:
            x = l(x)
            x = self.activation(x)
            feature_map.append(x)
        x = self.postconv(x)
        feature_map.append(x)
        return torch.flatten(x, 1, -1), feature_map


class MPD(nn.Module):
    def __init__(self, periods=[2,3,5,7,11]):
        super(MPD, self).__init__()
        self.subdiscriminators = nn.ModuleList()
        for p in periods:
            self.subdiscriminators.append(PeriodDiscriminator(p))

    def forward(self, x):
        preds, feature_maps = [], []
        for discriminator in self.subdiscriminators:
            pred, fm = discriminator(x)
            preds.append(pred)
            feature_maps.append(fm)
        return preds, feature_maps


class MSD(nn.Module):
    def __init__(self, stages=3):
        super(MSD, self).__init__()
        self.subdiscriminators = nn.ModuleList()
        self.poolings = nn.ModuleList()
        self.subdiscriminators.append(ScaleDiscriminator(spectral_norm))
        for _ in range(stages-1):
            self.subdiscriminators.append(nn.Sequential([
                nn.AvgPool1d(4,2,2),
                ScaleDiscriminator(weight_norm)
            ]))

    def forward(self, x):
        preds, feature_maps = [], []
        for discriminator in self.subdiscriminators:
            pred, fm = discriminator(x)
            preds.append(pred)
            feature_maps.append(fm)
        return preds, feature_maps


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.mpd = MPD()
        self.msd = MSD()

    def forward(self, x):
        return *self.mpd(x), *self.msd(x)
