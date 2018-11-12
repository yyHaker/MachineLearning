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
from data_helper import load_sample_data, load_mnist_data


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

    def train(self, datas, another=3):
        """
        :param datas: datas
        :param another: another class from idx.
        :return:
        """
        # 增广数据
        # n x d
        n, d = datas.shape
        # aug datas (n, d) -> (n, 1+d)
        ones = np.ones(n)
        datas = np.insert(datas, 0, values=ones, axis=1)

        # 第二种类别乘以-1
        datas[another:] = datas[another:] * (-1)

        # 伪逆求解
        b = np.ones(n)
        YTY = np.dot(datas.T, datas)
        inv = np.linalg.inv(YTY)
        false_inv = np.dot(inv, datas.T)
        alpha = np.dot(false_inv, b)
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
    print("last alpha: ", alpha)
    lsme.plot_result(datas, alpha)


def experiment2():
    """
    使用LSME算法分类多类样本.
    ------
    1. 数据集归类
    2. 每两类训练一个感知器模型，得到参数alpha，这样总共得到c(c-1)/2个
    3. 对于测试样本x, 如果g_{ij}(x)>=0, 对于任意j不等于i，则x属于w_i.
    :return:
    """
    # 数据集归类
    train_datas, train_labels = load_mnist_data()
    n, d = train_datas.shape
    print("total train datas is {}, dim is: {}".format(n, d))
    datas = {}
    for idx, j in enumerate(train_labels):
        if j not in datas.keys():
            datas[j] = []
        datas[j].append(train_datas[idx])
    c = len(datas.keys())

    # 每两类训练一个感知器模型
    alphas = np.zeros((c, c, 1+d))
    combines = []
    for i in datas.keys():
        for j in datas.keys():
            if i != j:
                combines.append((i, j))
    for i, j in combines:
        dataij = np.concatenate((datas[i], datas[j]), axis=0)
        # print(len(dataij), len(datas[i]), len(datas[j]))
        lsme = LSME()
        alphas[i, j] = lsme.train(dataij, another=len(datas[i]))
        print("=========>>> alpha[{}. {}] is {}".format(i, j, alphas[i, j]))

    # 对于测试样本x, 如果g_{ij}(x)>=0, 对于任意j不等于i，则x属于w_i.
    test_datas, test_labels = load_mnist_data(filename='./data/TestSamples.csv',
                                              label_filename='./data/TestLabels.csv')
    test_n, test_dim = test_datas.shape
    print("total test datas is: {}, dim is: {}".format(test_n, test_dim))
    count = 0
    test_datas = np.insert(test_datas, 0, values=np.ones(test_n), axis=1)
    print("begin testing....")
    for idx, data in enumerate(test_datas):
        p_lbl = -1
        for i in range(c):
            flag = True
            for j in range(c):
                if (i, j) in combines:
                    if np.dot(data, alphas[i, j]) < 0:
                        flag = False
                        break
            if flag:
                p_lbl = i
        if p_lbl == test_labels[idx]:
            count += 1
    print("total test acc is {} / {} = {}".format(count, test_n, (count + 0.0)/test_n))


if __name__ == "__main__":
    experiment2()
