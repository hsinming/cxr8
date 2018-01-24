#!/usr/bin/python3
import torch
import torch.nn as nn
import torchvision.models as models
from se_module import SELayer
import math

se_resnet50_weight = './se_resnet50_weight.pkl'

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
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
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def se_resnet50(num_classes=1000):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = models.ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class ResNet50(nn.Module):
    def __init__(self, n_classes, logger=None):
        super(ResNet50, self).__init__()
        self.n_class = n_classes
        self.logger=logger

        # Get features from resnet50
        self.model_ft = models.resnet50(pretrained=True)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, self.n_class),
            nn.Sigmoid()
        )

    def log(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)  # out: bs*2048*32*32

        x = self.transition(x)  # out: bs*2048*32*32
        x = self.globalPool(x)  # out: bs*2048*1*1
        x = x.view(x.size(0), -1)  # out: bs*2048
        x = self.prediction(x)
        return x


class ResNet50_Avgpool(nn.Module):
    def __init__(self, n_classes, logger=None):
        super(ResNet50_Avgpool, self).__init__()
        self.n_class = n_classes
        self.logger=logger

        # Get features from resnet50
        self.model_ft = models.resnet50(pretrained=True)

        for param in self.model_ft.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.AvgPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, self.n_class),
            nn.Sigmoid()
        )

    def log(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def forward(self, x):
        x = self.model_ft.conv1(x)
        x = self.model_ft.bn1(x)
        x = self.model_ft.relu(x)
        x = self.model_ft.maxpool(x)

        x = self.model_ft.layer1(x)
        x = self.model_ft.layer2(x)
        x = self.model_ft.layer3(x)
        x = self.model_ft.layer4(x)  # out: bs*2048*32*32

        x = self.transition(x)  # out: bs*2048*32*32
        x = self.globalPool(x)  # out: bs*2048*1*1
        x = x.view(x.size(0), -1)  # out: bs*2048
        x = self.prediction(x)
        return x


class SE_ResNet50(nn.Module):
    def __init__(self, n_classes, logger=None):
        super(SE_ResNet50, self).__init__()
        self.n_class = n_classes
        self.logger=logger
        self.model = se_resnet50(self.n_class)

        # Load pretrained weight provided by author. (https://github.com/moskomule/senet.pytorch)
        pretrained_dict = torch.load(se_resnet50_weight)['weight']
        model_dict = self.model.state_dict()
        # Filter the unmatched keys
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

        for param in self.model.parameters():
            param.requires_grad = False

        self.transition = nn.Sequential(
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1, stride=1, bias=False),
        )
        self.globalPool = nn.Sequential(
            nn.MaxPool2d(32)
        )
        self.prediction = nn.Sequential(
            nn.Linear(2048, self.n_class),
            nn.Sigmoid()
        )

    def log(self, msg):
        if self.logger:
            self.logger.debug(msg)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # out: bs*2048*32*32

        x = self.transition(x)  # out: bs*2048*32*32
        x = self.globalPool(x)  # out: bs*2048*1*1
        x = x.view(x.size(0), -1)  # out: bs*2048
        x = self.prediction(x)
        return x
