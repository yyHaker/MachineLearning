# -*- coding: utf-8 -*-
import numpy as np
from data_helper import load_sample_data, load_mnist_data
import matplotlib.pyplot as plt


class Perceptron(object):
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

    def isMisClassified(self, data, alpha):
        """
        判断感知器是否将数据错误分类.
        :param data: 增广后的数据.
        :param alpha: alpha
        :return:  True or False.
        """
        res = np.dot(alpha, data)
        if res > 0:
            return False
        else:
            return True

    def train(self, datas, eta=1., theta=1e-4, another=3, mode='single'):
        """
        train the perceptron.
        :param datas: numpy array, (n, d)
        :param eta:
        :param theta:
        :param another: another class from idx.
        :param mode: 'single' or 'batch'
        :return:
           'alpha': numpy array, (1+d)
        """
        # nxd
        n, d = datas.shape
        # aug datas (n, d) -> (n, 1+d)
        ones = np.ones(n)
        datas = np.insert(datas, 0, values=ones, axis=1)
        # 第二种类别乘以-1
        datas[another:] = datas[another:] * (-1)

        # initialize alpha, eta, theta
        b = np.ones(d)
        alpha = np.concatenate(([0.], b))
        step = np.array(np.inf * (1+d))
        # print("init alpha: ", alpha)

        # 单样本调整版本的感知器算法
        if mode == 'single':
            while True:
                count = 0
                for idx, data in enumerate(datas):
                    if self.isMisClassified(data, alpha):
                        step = eta * data
                        alpha = alpha + step
                        # print("current alpha: ", alpha)
                    else:
                        count += 1
                # 所有的样本正确分类
                if count == n:
                    break
        # 批量调整版本的感知器算法
        elif mode == 'batch':
            while True:
                error_samples = np.zeros((1+d))
                for idx, data in enumerate(datas):
                    # 收集错误分类的样本
                    if self.isMisClassified(data, alpha):
                        error_samples += data
                # print("error samples: ", error_samples)
                # calc step
                step = eta * error_samples
                # update params
                alpha += step
                print("current step: ", step, "alpha: ", alpha)
                # 收敛条件
                if np.linalg.norm(step, ord=1) < theta:
                    break
        return alpha

    def plot_result(self, datas, alpha):
        """
        plot the res.
        :param datas: n x d.
        :param alpha: 1+d
        :return:
        """
        colorMark = ['or', 'dg']
        for idx, data in enumerate(datas):
            plt.plot(data[0], data[1], colorMark[idx >= 3])
        x = np.linspace(-1, 3, 1000)
        y = -(alpha[0] + alpha[1]*x) / alpha[2]
        plt.plot(x, y, color='blue')
        plt.show()


def experiment1():
    """
     单样本调整版本的感知器算法.
    :return:
    """
    datas, labels = load_sample_data()
    perceptron = Perceptron()
    alpha = perceptron.train(datas, labels)
    print("last alpha: ", alpha)
    perceptron.plot_result(datas, alpha)


def experiment2():
    """
    批量调整版本的感知器算法.
    :return:
    """
    datas, labels = load_sample_data()
    perceptron = Perceptron()
    alpha = perceptron.train(datas, labels, eta=0.01, theta=1e-4, mode='batch')
    print("last alpha: ", alpha)
    perceptron.plot_result(datas, alpha)


def experiment3():
    """
    使用感知器算法分类多类样本.
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
        perceptron = Perceptron()
        alphas[i, j] = perceptron.train(dataij, another=len(datas[i]))
        print("=========>>> alpha[i. j] is {}".format(alphas[i, j]))

    # 对于测试样本x, 如果g_{ij}(x)>=0, 对于任意j不等于i，则x属于w_i.
    test_datas, test_labels = load_mnist_data(filename='./data/TestSamples.csv',
                                             label_filename='./data/TestLabels.csv')
    test_n, test_dim = test_datas.shape
    print("total test datas is: {}, dim is: {}".format(test_n, test_dim))
    count = 0
    test_datas = np.insert(test_datas, 0, values=np.ones(test_n), axis=1)
    perceptron = Perceptron()
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
    experiment3()


