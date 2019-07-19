#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: activation_functions.py
@time: 2019/7/19 20:38
"""
import numpy as np


def sigmoid(x):
    """sigmoid function(vectorized)"""
    s = 1 / (1 + np.exp(-x))
    return s


def tanh(x):
    """tanh function(vectorized)"""
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


def relu(x):
    """relu function(vectorized)"""
    s = np.where(x < 0, 0, x)
    return s


def softmax(x):
    """softmax function(vectorized)"""
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

