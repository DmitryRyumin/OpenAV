#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Архитектуры нейросетевых моделей
"""

# ######################################################################################################################
# Импорт необходимых инструментов
# ######################################################################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ######################################################################################################################
# Модели
# ######################################################################################################################


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()

        self.i_downsample = i_downsample

        if self.i_downsample is not None:
            self.conv1 = Conv2dSame(in_channels, out_channels, 3, stride=stride, groups=1, bias=True)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding="same", bias=True)
        self.batch_norm1 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same", bias=True)
        self.batch_norm2 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.99)

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)

        return x


class Conv2dSame(torch.nn.Conv2d):
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv_layer_s2_same = Conv2dSame(num_channels, 64, 7, stride=2, groups=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64, stride=1)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def extract_features(self, x):
        x = self.relu(self.batch_norm1(self.conv_layer_s2_same(x)))
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes:
            ii_downsample = nn.Sequential(
                Conv2dSame(self.in_channels, planes, 1, stride=stride, groups=1, bias=True),
                nn.BatchNorm2d(planes, eps=0.001, momentum=0.99),
            )

        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes

        for i in range(blocks - 1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet18(channels=3):
    return ResNet(Bottleneck, [2, 2, 2, 2], channels)


class CNNLSTMPyTorch(nn.Module):
    def __init__(self, h_u=512, channels=3, p_drop=0.2):
        super(CNNLSTMPyTorch, self).__init__()
        self.resnet = ResNet18(channels=channels)
        self.lstm1 = nn.LSTM(input_size=h_u, hidden_size=h_u, batch_first=True, bidirectional=False)
        self.drop1 = nn.Dropout(p=p_drop)
        self.lstm2 = nn.LSTM(input_size=h_u, hidden_size=h_u, batch_first=True, bidirectional=False)
        self.drop2 = nn.Dropout(p=p_drop)

    def forward(self, x):
        x_n = rearrange(x, "b g c h w-> (b g) c h w")
        x_n = self.resnet(x_n)
        x_n = rearrange(x_n, "(b g) f-> b g f", b=x.shape[0], g=x.shape[1])
        x_n, _ = self.lstm1(x_n)
        x_n = self.drop1(x_n)
        x_n, _ = self.lstm2(x_n)
        x_n = x_n[:, -1, :]
        x_n = self.drop2(x_n)
        return x_n


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x1, x2, x3):
        queries = self.query(x1)
        keys = self.key(x2)
        values = self.value(x3)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim**0.5)
        attention = self.softmax(scores)
        weighted = torch.bmm(attention, values)
        return weighted


class InitialEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super(InitialEncoder, self).__init__()
        self.att = Attention(input_dim)
        self.lambda_ = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x1):
        x_w = self.att(x1, x1, x1)
        x_w = x1 * self.lambda_ + x_w * (1 - self.lambda_)
        x_w = self.ln1(x_w)
        x_w_fc1 = self.fc1(x_w)
        x_w = self.ln2(x_w + x_w_fc1)
        x_w = self.fc2(x_w)
        return x_w


class SecondEncoder(nn.Module):
    def __init__(self, input_dim=512):
        super(SecondEncoder, self).__init__()
        self.att1 = Attention(input_dim)
        self.att2 = Attention(input_dim)
        self.lambda_ = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)

    def forward(self, x1, x2):
        x_w1 = self.att1(x1, x1, x1)
        x_w2 = self.att2(x1, x2, x2)
        x_w12 = x_w1 * self.lambda_ + x_w2 * (1 - self.lambda_)
        x_w12 = self.ln1(x1 + x_w12)
        x_w12_fc1 = self.fc1(x_w12)
        x_w12 = self.ln2(x_w12 + x_w12_fc1)
        x_w12 = self.fc2(x_w12)
        return x_w12


class Decoder(nn.Module):
    def __init__(self, input_dim=512, h_u=512, n_class=500):
        super(Decoder, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(input_dim, h_u)
        self.fc2 = nn.Linear(h_u, n_class)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.avgpool(x).transpose(1, 2)
        x = x.reshape(x.shape[0], -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim=512, h_u=512, n_class=500, encoder_decoder=5):
        super(Transformer, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Создание энкодеров
        self.encoders.append(InitialEncoder(input_dim=input_dim))
        for _ in range(encoder_decoder - 1):
            self.encoders.append(SecondEncoder(input_dim=input_dim))

        # Создание декодеров
        for _ in range(encoder_decoder):
            self.decoders.append(Decoder(input_dim=input_dim, h_u=h_u, n_class=n_class))

    def forward(self, x):
        x_w = [x]

        x_w.append(self.encoders[0](x))

        for encoder in self.encoders[1:]:
            x_w.append(encoder(x, x_w[-1]))

        y_all = []
        for i, decoder in enumerate(self.decoders):
            y_all.append(decoder(x_w[i]))

        y_all = torch.stack(y_all)
        y_mean = torch.mean(y_all, dim=0)

        return y_mean


class AVModel(nn.Module):
    def __init__(self, shape_audio, shape_video, input_dim=512, h_u=512, h_f=64, n_class=500, encoder_decoder=5):
        super(AVModel, self).__init__()
        self.feature_audio = ResNet18(channels=shape_audio[2])
        self.fc_audio = nn.Linear(shape_audio[1], h_f)
        self.feature_video = CNNLSTMPyTorch()
        self.fc_video = nn.Linear(shape_video[1], h_f)
        self.fusion = Transformer(input_dim=input_dim, h_u=h_u, n_class=n_class, encoder_decoder=encoder_decoder)

    def forward(self, audio, video):
        audio_n = rearrange(audio, "b g c n l-> (b g) c n l")
        audio_n = rearrange(self.feature_audio(audio_n), "(b g) f -> b g f", b=audio.shape[0], g=audio.shape[1])
        audio_n = self.fc_audio(audio_n.transpose(1, 2)).transpose(1, 2)
        video_n = rearrange(video, "b g1 g2 c h w-> (b g1) g2 c h w")
        video_n = rearrange(self.feature_video(video_n), "(b g) f -> b g f", b=video.shape[0], g=video.shape[1])
        video_n = self.fc_video(video_n.transpose(1, 2)).transpose(1, 2)
        av = torch.cat([audio_n, video_n], dim=1)
        out = self.fusion(av)
        return out
