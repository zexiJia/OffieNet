import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)

def conv6x6(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=6, stride=stride, padding=0, bias=False)

def conv7x7(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3, bias=False)

class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, bn_mtm, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.bn_mtm = bn_mtm
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=self.bn_mtm)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=self.bn_mtm)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        return out


class ResNet_v2(nn.Module):

    def __init__(self, layers, width, bn_mtm):
        
        super(ResNet_v2, self).__init__()
        self.in_channels = width[0]
        self.bn_mtm = bn_mtm

        self.conv = conv7x7(1, width[0], 1)
        self.layer1 = self.make_layer(ResidualBlock, width[0], layers[0], 2)
        self.layer2 = self.make_layer(ResidualBlock, width[1], layers[1], 2)
        self.layer3 = self.make_layer(ResidualBlock, width[2], layers[2], 2)
        self.layer4 = self.make_layer(ResidualBlock, width[3], layers[3], 2)
        self.layer5 = self.make_layer(ResidualBlock, width[4], layers[4], 2)
        self.bn = nn.BatchNorm2d(width[4], momentum=self.bn_mtm)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AvgPool2d(3)

    def make_layer(self, block, out_channels, blocks, stride=1):
        
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels, momentum=self.bn_mtm))
        layers = []
        layers.append(block(self.in_channels, out_channels,
                            self.bn_mtm, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, self.bn_mtm))
        return nn.Sequential(*layers)

    def l2_norm(self, x):
        input_size = x.size()
        buffer_ = torch.pow(x, 2)
        normp = buffer_.sum(1, keepdim=True)
        normp = torch.add(normp, 1e-10)
        norm = torch.sqrt(normp)
        output = torch.div(x, norm)
        # output = _output.view(input_size)
        # output = torch.flatten
        return output

    def forward(self, x):
        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out_l = []

        out = self.layer5(out)
        out = self.bn(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        embedding1 = self.l2_norm(out)

        return embedding1