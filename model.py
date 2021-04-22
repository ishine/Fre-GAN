import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
import numpy as np
# import pywt
from pytorch_wavelets import DWT1DForward

LRELU_SLOPE = 0.1
mel_basis = {}
hann_window = {}

class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size, dilation):
        super(ResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[3],
                               padding=get_padding(kernel_size, dilation[3])))
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)

# For v3
class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1])))
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)

class Generator(torch.nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)  # always 3
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(h.num_mels, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2  # v3 만 ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(h.upsample_initial_channel // (2 ** i), h.upsample_initial_channel // (2 ** (i + 1)),
                                k, u, padding=(k - u) // 2)))

        self.up_conv = nn.ModuleList([
            weight_norm(Conv1d(1, 1, 3, 1, padding=1)),
            weight_norm(Conv1d(1, 1, 5, 1, padding=2)),
            weight_norm(Conv1d(1, 1, 9, 1, padding=4))
        ])

        self.skip_connect = nn.ModuleList([  # kernel,stride,padding # input mel len 이 32 인데 kernel 250은 너무 크자너
            weight_norm(ConvTranspose1d(h.num_mels, h.upsample_initial_channel // (2 ** 1), 10,
                                        np.prod(np.array(h.upsample_rates[:1])), padding=1)),
            weight_norm(ConvTranspose1d(h.num_mels, (h.upsample_initial_channel // (2 ** 2)), 36,
                                        np.prod(np.array(h.upsample_rates[:2])), padding=2)),
            weight_norm(ConvTranspose1d(h.num_mels, h.upsample_initial_channel // (2 ** 3), 72,
                                        np.prod(np.array(h.upsample_rates[:3])), padding=4)),
            weight_norm(ConvTranspose1d(h.num_mels, h.upsample_initial_channel // (2 ** 4), 144,
                                        np.prod(np.array(h.upsample_rates[:4])), padding=8)),
        ])

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))  # 하나 뒤 부터시작 // 128 --> 64 --> 32
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.sub_conv_post = nn.ModuleList()
        for i, u in enumerate(h.upsample_rates):
            if 0 < i:
                self.sub_conv_post.append(
                    weight_norm(Conv1d(h.upsample_initial_channel // (2 ** (i + 1)), 1, 7, 1, padding=3)))

        self.ups.apply(init_weights)
        self.up_conv.apply(init_weights)
        self.skip_connect.apply(init_weights)
        self.sub_conv_post.apply(init_weights)

    def forward(self, x):
        mel_condition = x
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            if i != 0:
                x = F.leaky_relu(x, LRELU_SLOPE)
                x += self.skip_connect[i - 1](mel_condition)
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)

            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

            if i == 1:
                wav2 = F.leaky_relu(x)
                wav2 = self.sub_conv_post[i - 1](wav2)
                wav2_out = torch.tanh(wav2) # wav
                wav2 = nn.Upsample(scale_factor=2, mode='nearest')(wav2_out)
                wav2 = self.up_conv[i - 1](wav2)  # + tanh?
                wav2 = torch.tanh(wav2)

            elif i == 2:
                wav6 = F.leaky_relu(x)
                wav6 = self.sub_conv_post[i - 1](wav6)
                wav6 = torch.tanh(wav6)
                wav6_out = wav2 + wav6
                wav6 = nn.Upsample(scale_factor=2, mode='nearest')(wav6_out)
                wav6 = self.up_conv[i - 1](wav6)
                wav6 = torch.tanh(wav6)

            elif i == 3:
                wav12 = F.leaky_relu(x)
                wav12 = self.sub_conv_post[i - 1](wav12)
                wav12 = torch.tanh(wav12)
                wav12_out = wav6 + wav12
                wav12 = nn.Upsample(scale_factor=2, mode='nearest')(wav12_out)
                wav12 = self.up_conv[i - 1](wav12)
                wav12 = torch.tanh(wav12)

            elif i == 4:
                # 24kHz
                wav24 = F.leaky_relu(x)
                wav24 = self.sub_conv_post[i - 1](wav24)
                wav24 = torch.tanh(wav24)
                wav24_out = wav12 + wav24
        return wav24_out

    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.conv_pre)

        for l in self.resblocks:
            l.remove_weight_norm()

        for l in self.ups:
            remove_weight_norm(l)
        for l in self.up_conv:
            remove_weight_norm(l)
        for l in self.skip_connect:
            remove_weight_norm(l)
        for l in self.sub_conv_post:
            remove_weight_norm(l)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, h, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.h = h
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.conv_pre = nn.ModuleList([
            norm_f(Conv2d(2, 32, (1, 1), (1, 1), padding=(0, 0))),  # [Batch, 1, 2048, 2] --> [Batch, 32, 2048, 2]
            norm_f(Conv2d(4, 128, (1, 1), (1, 1), padding=(0, 0))),  # [Batch, 1, 1024, 2] --> [Batch, 128, 1024, 2]
            norm_f(Conv2d(8, 512, (1, 1), (1, 1), padding=(0, 0)))  # [Batch, 1, 512, 2] --> [Batch, 256, 512, 2]

        ])

        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(32, 128, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(128, 512, (3, 1), (2, 1), padding=(get_padding(3, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])

        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        self.dwt2 = DWT1DForward(J=1, mode='zero', wave='db1') # /2, return wav, coeff[0]

        # self.merge_cnn = nn.ModuleLis([
        #     norm_f(Conv1d(2, 1, 1, 1, padding=0)),
        #     norm_f(Conv1d(4, 1, 1, 1, padding=0)),
        #     norm_f(Conv1d(8, 1, 1, 1, padding=0)),
        # ])

    def forward(self, x):
        x12_tmp, x12_coeff = self.dwt2(x)
        x12 = torch.cat((x12_tmp, x12_coeff[0]), dim=1)  # ch:2
        # x12 = self.merge_cnn[0](x12)

        x6_tmp, x6_coeff = self.dwt2(x12_tmp)
        x6_tmp2, x6_coeff2 = self.dwt2(x12_coeff[0])
        x6 = torch.cat((x6_tmp, x6_coeff[0], x6_tmp2, x6_coeff2[0]), dim=1)  # ch:4
        # x6 = self.merge_cnn[1](x6)

        x3_tmp, x3_coeff = self.dwt2(x6_tmp)
        x3_tmp2, x3_coeff2 = self.dwt2(x6_coeff[0])
        x3_tmp3, x3_coeff3 = self.dwt2(x6_tmp2)
        x3_tmp4, x3_coeff4 = self.dwt2(x6_coeff2[0])
        x3 = torch.cat((x3_tmp, x3_coeff[0], x3_tmp2, x3_coeff2[0], x3_tmp3, x3_coeff3[0], x3_tmp4, x3_coeff4[0]), dim=1)  # ch:8

        # x3 = self.merge_cnn[1](x3)

        # Reshape
        xes = []
        for xs in [x, x12, x6, x3]:
            b, c, t = xs.shape
            if t % self.period != 0:
                n_pad = self.period - (t % self.period)
                xs = F.pad(xs, (0, n_pad), "reflect")
                t = t + n_pad
            xes.append(xs.view(b, c, t // self.period, self.period))

        x, x12, x6, x3 = xes

        fmap = []
        for i, l in enumerate(self.convs):
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            if i < 3:
                fmap.append(x)
                res = self.conv_pre[i](xes[i + 1])
                res = F.leaky_relu(res, LRELU_SLOPE)
                x = (x + res) / torch.sqrt(torch.tensor(2.))
            else:
                fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiPeriodDiscriminator, self).__init__()
        self.h = h

        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, h),
            DiscriminatorP(3, h),
            DiscriminatorP(5, h),
            DiscriminatorP(7, h),
            DiscriminatorP(11, h),
        ])

    def forward(self, y, y_hat):  # y: FULL, y_hat: generated

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, h, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.h = h
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.conv_pre0 = nn.ModuleList([
            norm_f(Conv1d(2, 128, 1, 1, padding=0)),
            norm_f(Conv1d(4, 256, 1, 1, padding=0)),
            norm_f(Conv1d(8, 512, 1, 1, padding=0)),
        ])

        self.conv_pre1 = nn.ModuleList([
            norm_f(Conv1d(4, 128, 1, 1, padding=0)),
            norm_f(Conv1d(8, 256, 1, 1, padding=0)),
        ])

        self.conv_pre2 = nn.ModuleList([
            norm_f(Conv1d(8, 128, 1, 1, padding=0)),
        ])

        # in_channel, out_channel, kernel_size, st,
        self.convs0 = nn.ModuleList([  # input : [Batch, 1 , 8192]
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)), #4096-->2048
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 2, groups=16, padding=20)),  # strdie 4 --> 2
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 2, groups=16, padding=20)),  # stride 1 --> 2
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.convs1 = nn.ModuleList([  # input : [Batch, 1 , 8192]
            norm_f(Conv1d(2, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 2, groups=16, padding=20)),  # strdie 4 --> 2
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 2, groups=16, padding=20)),  # stride 1 --> 2
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.convs2 = nn.ModuleList([  # input : [Batch, 1 , 8192]
            norm_f(Conv1d(4, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 2, groups=16, padding=20)),  # strdie 4 --> 2
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 2, groups=16, padding=20)),  # stride 1 --> 2
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])

        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))
        self.dwt2 = DWT1DForward(J=1, mode='zero', wave='db1').cuda() # /2, return wav, coeff[0]

    def forward(self, x, num_dis):

        if num_dis == 0:
            x12_tmp, x12_coeff = self.dwt2(x)
            x12 = torch.cat((x12_tmp, x12_coeff[0]), dim=1) #ch:2

            x6_tmp, x6_coeff = self.dwt2(x12_tmp)
            x6_tmp2, x6_coeff2= self.dwt2(x12_coeff[0])
            x6 = torch.cat((x6_tmp, x6_coeff[0], x6_tmp2, x6_coeff2[0]), dim=1) #ch:4

            x3_tmp, x3_coeff = self.dwt2(x6_tmp)
            x3_tmp2, x3_coeff2 = self.dwt2(x6_coeff[0])
            x3_tmp3, x3_coeff3 = self.dwt2(x6_tmp2)
            x3_tmp4, x3_coeff4 = self.dwt2(x6_coeff2[0])
            x3 = torch.cat((x3_tmp, x3_coeff[0], x3_tmp2, x3_coeff2[0], x3_tmp3, x3_coeff3[0], x3_tmp4, x3_coeff4[0] ), dim=1) #ch:8
            xes = [x12, x6, x3]
            fmap = []
            for i, l in enumerate(self.convs0):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2, 3]:
                    fmap.append(x)
                    res = self.conv_pre0[i - 1](xes[i - 1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        elif num_dis == 1:
            x6_tmp, x6_coeff = self.dwt2(x)
            x6 = torch.cat((x6_tmp, x6_coeff[0]), dim=1) # ch:4, T:2048
            x3_tmp, x3_coeff = self.dwt2(x6_tmp)
            x3_tmp2, x3_coeff2 = self.dwt2(x6_coeff[0])
            x3 = torch.cat((x3_tmp, x3_coeff[0], x3_tmp2, x3_coeff2[0]), dim=1) # ch:8

            xes = [x6, x3]
            fmap = []
            for i, l in enumerate(self.convs1):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i in [1, 2]:
                    fmap.append(x)
                    res = self.conv_pre1[i-1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)
            return x, fmap

        else:
            x3_tmp, x3_coeff = self.dwt2(x)
            x3 = torch.cat((x3_tmp, x3_coeff[0]), dim=1) #ch:8
            xes = [x3]

            fmap = []
            for i, l in enumerate(self.convs2):
                x = l(x)
                x = F.leaky_relu(x, LRELU_SLOPE)
                if i == 1:
                    fmap.append(x)
                    res = self.conv_pre2[i - 1](xes[i-1])
                    res = F.leaky_relu(res, LRELU_SLOPE)
                    x = (x + res) / torch.sqrt(torch.tensor(2.))
                else:
                    fmap.append(x)

            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

            return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, h):
        super(MultiScaleDiscriminator, self).__init__()
        self.h = h

        self.discriminators = nn.ModuleList([
            DiscriminatorS(h, use_spectral_norm=True),
            DiscriminatorS(h),
            DiscriminatorS(h),
        ])

        self.dwt2 = DWT1DForward(J=1, mode='zero', wave='db1') # /2, return wav, coeff[0]


    def forward(self, y, y_hat):  # y: FULL, y_hat: generated

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        y2, y2_coeff = self.dwt2(y)
        y_down2 = torch.cat((y2, y2_coeff[0]), dim=1) # [B,2,T/2]

        yhat2, yhat2_coeff = self.dwt2(y_hat)
        yhat_down2 = torch.cat((yhat2, yhat2_coeff[0]), dim=1)  # [B,2,T/2]
        ###########
        y4_1, y4_1_coeff = self.dwt2(y2)
        y4_2, y4_2_coeff = self.dwt2(y2_coeff[0])
        y_down4 = torch.cat((y4_1, y4_1_coeff[0], y4_2, y4_2_coeff[0]), dim=1) #[B,4,T/4]

        yhat4_1, yhat4_1_coeff = self.dwt2(yhat2)
        yhat4_2, yhat4_2_coeff = self.dwt2(yhat2_coeff[0])
        yhat_down4 = torch.cat((yhat4_1, yhat4_1_coeff[0], yhat4_2, yhat4_2_coeff[0]), dim=1)  # [B,4,T/4]


        for i, d in enumerate(self.discriminators):
            # i: number of discriminator
            if i == 0:
                y_d_r, fmap_r = d(y, i)
                y_d_g, fmap_g = d(y_hat, i)
                y_d_rs.append(y_d_r)
                fmap_rs.append(fmap_r)
                y_d_gs.append(y_d_g)
                fmap_gs.append(fmap_g)

            elif i == 1:
                y_d_r, fmap_r = d(y_down2, i)
                y_d_g, fmap_g = d(yhat_down2, i)
                y_d_rs.append(y_d_r)
                fmap_rs.append(fmap_r)
                y_d_gs.append(y_d_g)
                fmap_gs.append(fmap_g)

            else:
                y_d_r, fmap_r = d(y_down4, i)
                y_d_g, fmap_g = d(yhat_down4, i)
                y_d_rs.append(y_d_r)
                fmap_rs.append(fmap_r)
                y_d_gs.append(y_d_g)
                fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))  # abs 말고 cos sim?

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg ** 2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses

