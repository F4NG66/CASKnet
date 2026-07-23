# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class Config(object):
    """配置参数"""

    def __init__(self, dataset, pretrained_name_or_path=None):
        self.dropout = 0.5  # 随机失活
        self.num_classes = 2  # 类别数
        self.pad_size = 32  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.embed = 1024
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(25, config.embed_size)
        self.embedding.weight.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embedding(x)
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out