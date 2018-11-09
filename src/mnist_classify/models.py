#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: models.py
@time: 2018/10/16 18:50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayersNetwork(nn.Module):
    """
    Multi-layers Network.
    """
    def __init__(self, n_features, hidden_1, hidden_2,  n_output):
        super(MultiLayersNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_features, hidden_1)  # hidden layer1
        self.hidden2 = nn.Linear(hidden_1, hidden_2)
        self.output = nn.Linear(hidden_2, n_output)   # output layer

    def forward(self, x):
        x = F.dropout(F.relu(self.hidden1(x)))
        x = F.dropout(F.relu(self.hidden2(x)))
        x = self.output(x)
        return x
