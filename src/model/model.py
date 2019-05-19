import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from .. import config
from ..delf.layers import SpatialAttention2d, WeightedSum2d
from .mobilev2 import MobileNetV2, InvertedResidual
# import matplotlib.pyplot as plt
# import numpy as np


"""
def imshow(img):
    # print(type(img))
    img = img * 0.23 + 0.5     # unnormalize
    npimg = img.numpy()
    # print(npimg.shape)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
"""


trn_trnsfms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.RandomCrop(config.INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

tst_trnsfms = transforms.Compose([
    transforms.Resize(config.IMG_SIZE),
    transforms.CenterCrop(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


class ResNet(nn.Module):
    def __init__(self, output_neurons, n_classes, dropout_rate):
        super(ResNet, self).__init__()
        # self.resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torchvision.models.resnet34(pretrained=True)
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        n_out_channels = 512  # resnet18, 34: 512, resnet50: 512*4
        self.norm1 = nn.BatchNorm1d(n_out_channels)
        self.drop1 = nn.Dropout(dropout_rate)
        # FC
        self.fc = nn.Linear(n_out_channels, output_neurons)
        self.norm2 = nn.BatchNorm1d(output_neurons)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        # x = self.norm2(x)
        # x = l2_norm(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, output_neurons, n_classes, dropout_rate):
        super(DenseNet, self).__init__()
        self.densenet_features = torchvision.models.densenet121(
            pretrained=True).features
        self.norm1 = nn.BatchNorm1d(1024)
        self.drop1 = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(1024, output_neurons)
        self.norm2 = nn.BatchNorm1d(output_neurons)

    def forward(self, x):
        features = self.densenet_features(x)
        x = F.relu(features, inplace=True)
        # GAP
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        x = self.norm1(x)
        x = self.drop1(x)
        # FC
        x = self.fc(x)
        x = self.norm2(x)
        return x


class DelfResNet(nn.Module):
    def __init__(self):
        super(DelfResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # self.resnet = torchvision.models.resnet34(pretrained=True)
        # self.resnet = torchvision.models.resnet18(pretrained=True)
        d_delf = config.latent_dim
        self.conv = nn.Conv2d(1024, d_delf, 1, 1)
        self.attn = SpatialAttention2d(in_c=d_delf, act_fn='relu')
        self.pool = WeightedSum2d()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        # x = self.resnet.layer4(x)
        x = self.conv(x)
        # x = F.relu(x)
        # print('conv: %s' % str(x.size()))

        attn_x = F.normalize(x, p=2, dim=1)
        attn_score = self.attn(x)
        x = self.pool([attn_x, attn_score])
        x = x.view(x.size(0), -1)

        return x


def pool_attn(attn):
    """
    pooling attention score to 1/2
    """
    assert (attn.size(2) % 2 == 0) and (attn.size(3) % 2 == 0)
    weight = torch.ones(1, 1, 2, 2).cuda()
    out = F.conv2d(attn, weight, stride=2)

    return out


class DelfMoblileNetV2(nn.Module):
    def __init__(self):
        super(DelfMoblileNetV2, self).__init__()
        self.features = MobileNetV2(
            n_class=1, input_size=config.INPUT_SIZE[0]).features

        d_delf = config.latent_dim
        d_half = round(d_delf / 2)

        self.inv_conv1 = InvertedResidual(96, d_half, 1, 0.5)
        self.inv_conv2 = InvertedResidual(1280, d_half, 1, 0.2)

        self.attn1 = SpatialAttention2d(in_c=d_half, act_fn='relu')
        self.attn2 = SpatialAttention2d(in_c=d_half, act_fn='relu')
        self.pool = WeightedSum2d()

    def forward(self, x):
        for layer in self.features[:14]:
            x = layer(x)
        # print('features size: %s' % str(x.size()))  # 96x14x14

        x1 = self.inv_conv1(x)

        attn_x = F.normalize(x1, p=2, dim=1)
        attn_score = self.attn1(x1)
        x1 = self.pool([attn_x, attn_score])
        x1 = x1.view(x1.size(0), -1)

        for layer in self.features[14:]:
            x = layer(x)
        # print('features size: %s' % str(x.size()))  # 1280x7x7

        x2 = self.inv_conv2(x)

        attn_x = F.normalize(x2, p=2, dim=1)
        attn_score = pool_attn(attn_score)
        x2 = self.pool([attn_x, attn_score])
        x2 = x2.view(x2.size(0), -1)

        x = torch.cat([x1, x2], 1)

        return x

    """
    def forward(self, x):
        x = self.features(x)
        # print(self.features)
        # print('features size: %s' % str(x.size()))

        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        attn_x = F.normalize(x, p=2, dim=1)
        attn_score = self.attn(x)
        x = self.pool([attn_x, attn_score])
        x = x.view(x.size(0), -1)

        return x
    """
