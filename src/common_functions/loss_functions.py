#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: loss_functions.py
@time: 2019/7/19 20:29
"""
"""
实现常见的损失函数
"""
import numpy as np


def cross_entropy(Y, P):
    """Cross-Entropy loss function.
    以向量化的方式实现交叉熵函数
    Y and P are lists of labels and estimations
    returns the float corresponding to their cross-entropy.
    """
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P)) / len(Y)


