import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from .. import config
from ..delf.layers import SpatialAttention2d, WeightedSum2d
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
