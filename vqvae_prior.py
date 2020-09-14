import matplotlib.pyplot as plt

from __future__ import print_function

import numpy as np

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from tqdm import tqdm

from math import sqrt

from vqvae import VQVAE


def preprocess(x, n_bits):
    #     """ preprosses discrete latents space [0, 2**n_bits) to model space [-1,1]; if size of the codebook ie n_embeddings = 512 = 2**9 -> n_bit=9 """
    # 1. convert data to float
    # 2. normalize to [0,1] given quantization
    # 3. shift to [-1,1]
    return x.float().div(2 ** n_bits - 1).mul(2).add(-1)


def deprocess(x, n_bits):
    #    """ deprocess x from model space [-1,1] to discrete latents space [0, 2**n_bits) where 2**n_bits is size of the codebook """
    # 1. shift to [0,1]
    # 2. quantize to n_bits
    # 3. convert data to long
    return x.add(1).div(2).mul(2 ** n_bits - 1).long()


# ==============
# PixelSNAIL top prior
# ==============


def down_shift(x):
    return F.pad(x, (0, 0, 1, 0))[:, :, :-1, :]


def right_shift(x):
    return F.pad(x, (1, 0))[:, :, :, :-1]


def concat_elu(x):
    return F.elu(torch.cat([x, -x], dim=1))


class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.utils.weight_norm(self)


class DownConv(Conv2d):
    def forward(self, x):
        Hk, Wk = self.kernel_size
        x = F.pad(x, ((Wk - 1) // 2, (Wk - 1) // 2, Hk - 1, 0))
        return super().forward(x)


class DownRightConv(Conv2d):
    def forward(self, x):
        Hk, Wk = self.kernel_size
        x = F.pad(x, (Wk - 1, 0, Hk - 1, 0))
        return super().forward(x)


class GatedResLayer(nn.Module):
    def __init__(self, conv, n_channels, kernel_size, drop_rate=0, shortcut_channels=None, n_cond_classes=None, relu_fn=concat_elu):
        super().__init__()
        self.relu_fn = relu_fn

        self.c1 = conv(2*n_channels, n_channels, kernel_size)
        if shortcut_channels:
            self.c1c = Conv2d(2*shortcut_channels, n_channels, kernel_size=1)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.c2 = conv(2*n_channels, 2*n_channels, kernel_size)
        if n_cond_classes:
            self.proj_y = nn.Linear(n_cond_classes, 2*n_channels)

    def forward(self, x, a=None, y=None):
        c1 = self.c1(self.relu_fn(x))
        if a is not None:  # shortcut connection if auxiliary input 'a' is given
            c1 = c1 + self.c1c(self.relu_fn(a))
        c1 = self.relu_fn(c1)
        if hasattr(self, 'dropout'):
            c1 = self.dropout(c1)
        c2 = self.c2(c1)
        if y is not None:
            c2=c2.transpose(1,3)
            c2 += self.proj_y(y)[:,:,None]
            c2=c2.transpose(1,3)
        a, b = c2.chunk(2,1)
        out = x + a * torch.sigmoid(b)
        return out


def causal_attention(k, q, v, mask, nh, drop_rate, training):
    B, dq, H, W = q.shape
    _, dv, _, _ = v.shape

    # split channels into multiple heads, flatten H,W dims and scale q; out (B, nh, dkh or dvh, HW)
    flat_q = q.reshape(B, nh, dq // nh, H, W).flatten(3) * (dq // nh) ** -0.5
    flat_k = k.reshape(B, nh, dq // nh, H, W).flatten(3)
    flat_v = v.reshape(B, nh, dv // nh, H, W).flatten(3)

    logits = torch.matmul(flat_q.transpose(2, 3), flat_k)  # (B,nh,HW,dq) dot (B,nh,dq,HW) = (B,nh,HW,HW)
    logits = F.dropout(logits, p=drop_rate, training=training, inplace=True)
    logits = logits.masked_fill(mask == 0, -1e10)
    weights = F.softmax(logits, -1)

    attn_out = torch.matmul(weights, flat_v.transpose(2, 3))  # (B,nh,HW,HW) dot (B,nh,HW,dvh) = (B,nh,HW,dvh)
    attn_out = attn_out.transpose(2, 3)  # (B,nh,dvh,HW)
    return attn_out.reshape(B, -1, H, W)  # (B,dv,H,W)


class AttentionGatedResidualLayer(nn.Module):
    def __init__(self, n_channels, n_background_ch, n_res_layers, n_cond_classes, drop_rate, nh, dq, dv,
                 attn_drop_rate):
        super().__init__()
        # attn params
        self.nh = nh
        self.dq = dq
        self.dv = dv
        self.attn_drop_rate = attn_drop_rate

        self.input_gated_resnet = nn.ModuleList([
            *[GatedResLayer(DownRightConv, n_channels, (2, 2), drop_rate, None, n_cond_classes) for _ in
              range(n_res_layers)]])
        self.in_proj_kv = nn.Sequential(
            GatedResLayer(Conv2d, 2 * n_channels + n_background_ch, 1, drop_rate, None, n_cond_classes),
            Conv2d(2 * n_channels + n_background_ch, dq + dv, 1))
        self.in_proj_q = nn.Sequential(
            GatedResLayer(Conv2d, n_channels + n_background_ch, 1, drop_rate, None, n_cond_classes),
            Conv2d(n_channels + n_background_ch, dq, 1))
        self.out_proj = GatedResLayer(Conv2d, n_channels, 1, drop_rate, dv, n_cond_classes)

    def forward(self, x, background, attn_mask, y=None):
        ul = x
        for m in self.input_gated_resnet:
            ul = m(ul, y=y)

        kv = self.in_proj_kv(torch.cat([x, ul, background], 1))
        k, v = kv.split([self.dq, self.dv], 1)
        q = self.in_proj_q(torch.cat([ul, background], 1))
        attn_out = causal_attention(k, q, v, attn_mask, self.nh, self.attn_drop_rate, self.training)
        return self.out_proj(ul, attn_out)


class PixelSNAIL(nn.Module):
    def __init__(self, input_dims, n_channels, n_res_layers, n_out_stack_layers, n_cond_classes, n_bits,
                 attn_n_layers=4, attn_nh=8, attn_dq=16, attn_dv=128, attn_drop_rate=0, drop_rate=0.5, **kwargs):
        super().__init__()
        H, W = input_dims[2]
        # init background
        background_v = ((torch.arange(H, dtype=torch.float) - H / 2) / 2).view(1, 1, -1, 1).expand(1, 1, H, W)
        background_h = ((torch.arange(W, dtype=torch.float) - W / 2) / 2).view(1, 1, 1, -1).expand(1, 1, H, W)
        self.register_buffer('background', torch.cat([background_v, background_h], 1))
        # init attention mask over current and future pixels
        attn_mask = torch.tril(torch.ones(1, 1, H * W, H * W),
                               diagonal=-1).byte()  # 1s below diagonal -- attend to context only
        self.register_buffer('attn_mask', attn_mask)

        # input layers for `up` and `up and to the left` pixels
        self.ul_input_d = DownConv(2, n_channels, kernel_size=(1, 3))
        self.ul_input_dr = DownRightConv(2, n_channels, kernel_size=(2, 1))
        self.ul_modules = nn.ModuleList([
            *[AttentionGatedResidualLayer(n_channels, self.background.shape[1], n_res_layers, n_cond_classes, drop_rate,
                                          attn_nh, attn_dq, attn_dv, attn_drop_rate) for _ in range(attn_n_layers)]])
        self.output_stack = nn.Sequential(
            *[GatedResLayer(DownRightConv, n_channels, (2, 2), drop_rate, None, n_cond_classes) \
              for _ in range(n_out_stack_layers)])
        self.output_conv = Conv2d(n_channels, 2 ** n_bits, kernel_size=1)

    def forward(self, x, y=None):
        # add channel of ones to distinguish image from padding later on
        x = F.pad(x, (0, 0, 0, 0, 0, 1), value=1)

        ul = down_shift(self.ul_input_d(x)) + right_shift(self.ul_input_dr(x))
        for m in self.ul_modules:
            ul = m(ul, self.background.expand(x.shape[0], -1, -1, -1), self.attn_mask, y)
        ul = self.output_stack(ul)
        return self.output_conv(F.elu(ul)).unsqueeze(2)  # out (B, 2**n_bits, 1, H, W)


# =============
# PixelCNN bottom prior
# =============


def pixelcnn_gate(x):
    a, b = x.chunk(2, 1)
    return torch.tanh(a) * torch.sigmoid(b)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super().__init__(*args, **kwargs)

    def apply_mask(self):
        H, W = self.kernel_size
        self.weight.data[:, :, H // 2 + 1:, :].zero_()  # mask out rows below the middle
        self.weight.data[:, :, H // 2, W // 2 + 1:].zero_()  # mask out center row pixels right of middle
        if self.mask_type == 'a':
            self.weight.data[:, :, H // 2, W // 2] = 0  # mask out center pixel

    def forward(self, x):
        self.apply_mask()
        return super().forward(x)


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_cond_channels, drop_rate):
        super().__init__()
        self.residual = (in_channels == out_channels)
        self.drop_rate = drop_rate

        self.v = nn.Conv2d(in_channels, 2 * out_channels, kernel_size, padding=kernel_size // 2)  # vertical stack
        self.h = nn.Conv2d(in_channels, 2 * out_channels, (1, kernel_size),
                           padding=(0, kernel_size // 2))  # horizontal stack
        self.v2h = nn.Conv2d(2 * out_channels, 2 * out_channels, kernel_size=1)  # vertical to horizontal connection
        self.h2h = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)  # horizontal to horizontal

        if n_cond_channels:
            self.in_proj_y = nn.Conv2d(n_cond_channels, 2 * out_channels, kernel_size=1)

        if self.drop_rate > 0:
            self.dropout_h = nn.Dropout(drop_rate)

    def apply_mask(self):
        self.v.weight.data[:, :, self.v.kernel_size[0] // 2:, :].zero_()  # mask out middle row and below
        self.h.weight.data[:, :, :,
        self.h.kernel_size[1] // 2 + 1:].zero_()  # mask out to the right of the central column

    def forward(self, x_v, x_h, y):
        self.apply_mask()

        # projection of y if included for conditional generation (cf paper section 2.3 -- added before the pixelcnn_gate)
        proj_y = self.in_proj_y(y)

        # vertical stack
        x_v_out = self.v(x_v)
        x_v2h = self.v2h(x_v_out) + proj_y
        x_v_out = pixelcnn_gate(x_v_out)

        # horizontal stack
        x_h_out = self.h(x_h) + x_v2h + proj_y
        x_h_out = pixelcnn_gate(x_h_out)
        if self.drop_rate:
            x_h_out = self.dropout_h(x_h_out)
        x_h_out = self.h2h(x_h_out)

        # residual connection
        if self.residual:
            x_h_out = x_h_out + x_h

        return x_v_out, x_h_out

    def extra_repr(self):
        return 'residual={}, drop_rate={}'.format(self.residual, self.drop_rate)


class PixelCNN(nn.Module):
    def __init__(self, n_channels, n_out_conv_channels, kernel_size, n_res_layers, n_cond_stack_layers, n_cond_classes,
                 n_bits,
                 drop_rate=0, **kwargs):
        super().__init__()
        # conditioning layers (bottom prior conditioned on class labels and top-level code)
        self.in_proj_y = nn.Linear(n_cond_classes, 2 * n_channels)
        self.in_proj_h = nn.ConvTranspose2d(1, n_channels, kernel_size=4, stride=2,
                                            padding=1)  # upsample top codes to bottom-level spacial dim
        self.cond_layers = nn.ModuleList([
            GatedResLayer(partial(Conv2d, padding=kernel_size // 2), n_channels, kernel_size, drop_rate, None,
                          n_cond_classes) \
            for _ in range(n_cond_stack_layers)])
        self.out_proj_h = nn.Conv2d(n_channels, 2 * n_channels,
                                    kernel_size=1)  # double channels top apply pixelcnn_gate

        # pixelcnn layers
        self.input_conv = MaskedConv2d('a', 1, 2 * n_channels, kernel_size=7, padding=3)
        self.res_layers = nn.ModuleList([
            GatedResidualBlock(n_channels, n_channels, kernel_size, 2 * n_channels, drop_rate) for _ in
            range(n_res_layers)])
        self.conv_out1 = nn.Conv2d(n_channels, 2 * n_out_conv_channels, kernel_size=1)
        self.conv_out2 = nn.Conv2d(n_out_conv_channels, 2 * n_out_conv_channels, kernel_size=1)
        self.output = nn.Conv2d(n_out_conv_channels, 2 ** n_bits, kernel_size=1)

    def forward(self, x, h=None, y=None):
        # conditioning inputs -- h is top-level codes; y is class labels
        h = self.in_proj_h(h)
        for l in self.cond_layers:
            h = l(h, y=y)
        h = self.out_proj_h(h)
        y = self.in_proj_y(y)[:, :, None]
        y = y.transpose(1, 3)
        x = pixelcnn_gate(self.input_conv(x) + h + y)
        x_v, x_h = x, x

        for l in self.res_layers:
            x_v, x_h = l(x_v, x_h, y)
        out = pixelcnn_gate(self.conv_out1(x_h))
        out = pixelcnn_gate(self.conv_out2(out))
        return self.output(out).unsqueeze(2)  # (B, 2**n_bits, 1, H, W)