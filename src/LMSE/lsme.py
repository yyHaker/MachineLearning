#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: lsme.py
@time: 2018/11/11 18:52
"""

import numpy as np
import matplotlib.pyplot as plt
from data_helper import load_sample_data


class LSME(object):
    """最小均方误差算法"""
    def __init__(self):
        pass

    def predict(self, data, alpha):
        """
        predict the label  of the sample data.
        :param data: 增广后的数据
        :param alpha: alpha
        :return:
          'label': 1 or 2. (类别)
        """
        res = np.dot(alpha, data)
        if res > 0:
            return 1
        else:
            return 2

    def train(self, datas):
        # 增广数据
        # n x d
        n, d = datas.shape
        # aug datas (n, d) -> (n, 1+d)
        ones = np.ones(n)
        datas = np.insert(datas, 0, values=ones, axis=1)

        # 第二种类别乘以-1
        datas[3:] = datas[3:] * (-1)

        # 伪逆求解
        b = np.ones(n)
        YTY = np.dot(datas.T, datas)
        inv = np.linalg.inv(YTY)
        false_inv = np.dot(inv, datas.T)
        alpha = np.dot(false_inv, b)
        print("last alpha: ", alpha)
        return alpha

    def plot_result(self, datas, alpha):
        """
        plot the res.
        :param datas: n x d.
        :param alpha: 1+d.
        :return:
        """
        colorMark = ['or', 'dg']
        for idx, data in enumerate(datas):
            plt.plot(data[0], data[1], colorMark[idx >= 3])
        x = np.linspace(-1, 3, 1000)
        y = -(alpha[0] + alpha[1] * x) / alpha[2]
        plt.plot(x, y, color='blue')
        plt.show()


def experiment():
    datas, labels = load_sample_data()
    lsme = LSME()
    alpha = lsme.train(datas)
    lsme.plot_result(datas, alpha)


if __name__ == "__main__":
    experiment()
