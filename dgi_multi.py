"""
Deep Graph Infomax in DGL

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
"""

import torch
import torch.nn as nn
import math
from gcn import GCN

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout):
        super(Encoder, self).__init__()
        self.g = g
        self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        features = self.conv(features)
        return features


class MultiPooling(nn.Module):
    def __init__(self, pooling_methods):
        super(MultiPooling, self).__init__()
        self.pooling_methods = pooling_methods
        self.valid_pooling_methods = ['mean', 'max', 'min', 'sum', 'std', 'median', 'l2_norm', 'l1_norm']

    def forward(self, features):
        pooled_summaries = []
        for method in self.pooling_methods:
            if method not in self.valid_pooling_methods:
                raise ValueError(
                    f"Invalid pooling method: {method}. Please use one of the following: {self.valid_pooling_methods}")
            if not method:
                raise ValueError(f"No method selected.")
            if method == 'mean':
                pooled_summaries.append(torch.sigmoid(torch.mean(features, dim=0, keepdim=True)))
            elif method == 'max':
                pooled_summaries.append(torch.sigmoid(torch.max(features, dim=0)[0].unsqueeze(0)))
            elif method == 'min':
                pooled_summaries.append(torch.sigmoid(torch.min(features, dim=0)[0].unsqueeze(0)))
            elif method == 'sum':
                pooled_summaries.append(torch.sigmoid(torch.sum(features, dim=0).unsqueeze(0)))
            elif method == 'std':
                pooled_summaries.append(torch.sigmoid(torch.std(features, dim=0, unbiased=True).unsqueeze(0)))
            elif method == 'median':
                pooled_summaries.append(torch.sigmoid(torch.median(features, dim=0)[0].unsqueeze(0)))
            elif method == 'l2_norm':
                pooled_summaries.append(torch.sigmoid(torch.norm(features, p=2, dim=0, keepdim=True)))
            elif method == 'l1_norm':
                pooled_summaries.append(torch.sigmoid(torch.norm(features, p=1, dim=0, keepdim=True)))

            # Add more pooling methods here if needed.
        summary = torch.cat(pooled_summaries, dim=0)
        return summary


class Discriminator(nn.Module):
    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.reset_parameters()

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def reset_parameters(self):
        size = self.weight.size(0)
        self.uniform(size, self.weight)

    def forward(self, features, summary):
        features = torch.matmul(features, torch.matmul(self.weight, summary))
        return features


class Multi_DGI(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, pooling_methods):
        super(Multi_DGI, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout)
        self.discriminator = Discriminator(n_hidden)
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = nn.MSELoss()

        self.multi_pooling = MultiPooling(pooling_methods)
        self.attention_weights = nn.Parameter(torch.Tensor(len(pooling_methods), 1))  # 添加权重矩阵W
        self.reset_parameters()

    def reset_parameters(self):
        self.uniform(len(self.multi_pooling.pooling_methods), self.attention_weights)  # 初始化权重矩阵W

    def uniform(self, size, tensor):
        bound = 1.0 / math.sqrt(size)
        if tensor is not None:
            tensor.data.uniform_(-bound, bound)

    def forward(self, features):
        positive = self.encoder(features, corrupt=False)
        negative = self.encoder(features, corrupt=True)

        # 使用multi_pooling方法计算summary
        summary = self.multi_pooling(positive).t()

        positive_output = self.discriminator(positive, summary)
        negative_output = self.discriminator(negative, summary)

        # Apply softmax normalization to attention weights.
        normalized_weights = torch.softmax(self.attention_weights, dim=0)

        # Multiply the attention weights with the positive and negative outputs.
        positive_output = torch.matmul(positive_output, normalized_weights)
        negative_output = torch.matmul(negative_output, normalized_weights)

        l1 = self.loss(positive_output, torch.ones_like(positive_output))
        l2 = self.loss(negative_output, torch.zeros_like(negative_output))

        return l1 + l2


class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes, dropout=0.2):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_hidden, int(n_hidden//4)),
            nn.PReLU(),
            nn.Dropout(dropout),
            # nn.Linear(int(n_hidden//2), int(n_hidden//4)),
            # nn.PReLU(),
            # nn.Dropout(dropout),
            nn.Linear(int(n_hidden//4), n_classes)
        )
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features):
        features = self.fc(features)
        return features

