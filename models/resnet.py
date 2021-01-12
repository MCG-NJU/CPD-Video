"""
    Inflating 2D resnet to 3D following SlowOnly network in SlowFast.
    (https://arxiv.org/abs/1812.03982)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]


def conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, temporal_interact=False):
        super(BasicBlock, self).__init__()
        if temporal_interact:
            self.conv1 = conv3d(inplanes, planes, (3, 3, 3), stride, (1, 1, 1))
        else:
            self.conv1 = conv3d(inplanes, planes, (1, 3, 3), stride, (0, 1, 1))
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3d(planes, planes, (1, 3, 3), padding=(0, 1, 1))
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, temporal_interact=False):
        super(Bottleneck, self).__init__()
        if temporal_interact:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        else:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(1, 1, 1), padding=(0, 0, 0), bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=stride, padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=(1, 1, 1), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_size,
                 sample_duration,
                 dp,
                 shortcut_type='B',
                 num_classes=400,
                 extract_feature=False,
                 freeze_bn=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(1, 7, 7),
            # kernel_size=7,
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            # padding=(3, 3, 3),
            bias=False
            # bias=True
            )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, temporal_interact=False)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2, temporal_interact=False)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, temporal_interact=True)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, temporal_interact=True)

        last_duration = int(math.ceil(sample_duration / 1))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.avgpool_exf = nn.AvgPool3d(
            (last_duration, 1, 1), stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.dp = nn.Dropout(p=dp)
        self.extract_feature = extract_feature
        self.freeze_bn = freeze_bn
        print('this resnet has DROPOUT={}!'.format(dp))
        if self.extract_feature:
            print("resnet: EXTRACT FEATURE MODE")

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, temporal_interact=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=(1, stride, stride))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, (1, stride, stride), downsample, temporal_interact=temporal_interact))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, temporal_interact=temporal_interact))

        return nn.Sequential(*layers)

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        if self.freeze_bn:
            print("Freezing BatchNorm")
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()

    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)

        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.dp(x)
        x = self.fc(x)
        return x


def get_fine_tuning_parameters(model, ft_begin_index, lr):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')
    
    parameters = []
    for name, param in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in name:
                parameters.append({'params': param})
                print('update:', name)
                break
        else:
            param.requires_grad = False
    return parameters


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
