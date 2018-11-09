# -*- coding: utf-8 -*-
import numpy as np
from data_helper import load_sample_data, load_sample1_data


class Perceptron(object):
    def __init__(self):
        pass

    def predict(self, data, alpha):
        """
        predict the label  of the sample data.
        :param data:
        :param alpha:
        :return:
          'label': 1 or 0.
        """
        res = np.dot(alpha, data)
        if res > 0:
            return 1
        else:
            return 0

    def train(self, datas, labels, eta=0.01, theta=1e-4, mode='batch'):
        """
        train the perceptron.
        :param datas: numpy array, (n, d)
        :param labels: numpy array.
        :param eta:
        :param theta:
        :param mode: 'single' or 'batch'
        :return:
           'alpha': numpy array, (1+d)
        """
        n, d = datas.shape
        # aug datas (n, d) -> (n, 1+d)
        ones = np.ones(n)
        datas = np.insert(datas, 0, values=ones, axis=1)

        # initialize alpha, eta, theta
        b = np.ones(d)
        alpha = np.concatenate(([-4.], b))
        step = np.array(np.inf * (1+d))
        print("init alpha: ", alpha)

        if mode == 'single':
            while True:
                error_samples = np.zeros((1+d))
                for idx, data in enumerate(datas):
                    if self.predict(alpha, data) != labels[idx]:
                        step = eta * data
                        alpha = alpha + step
                        print("current alpha: ", alpha)
        elif mode == 'batch':
            while np.any(step >= theta):
                error_samples = np.zeros((1+d))
                for idx, data in enumerate(datas):
                    if self.predict(alpha, data) != labels[idx]:
                        # print("predict label: ", self.predict(data, alpha), " true label: ", labels[idx], "idx: ", idx)
                        error_samples += data
                print("error samples: ", error_samples)
                # calc step
                step = eta * error_samples
                # update params
                alpha += step
            print("current step: ", step, "alpha: ", alpha)
        return alpha


if __name__ == "__main__":
    datas, labels = load_sample1_data()
    perceptron = Perceptron()
    alpha = perceptron.train(datas, labels)
    print("last alpha: ", alpha)


