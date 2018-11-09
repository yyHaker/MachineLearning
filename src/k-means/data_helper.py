# -*- coding: utf-8 -*-
import csv
import random
import numpy as np


def load_iris_data(filename, split):
    """
    load iris data, and split the training and test set.
    :param filename: the data path
    :param split: split rate
    :return:
       'trainSet':
       'testSet':
    """
    trainSet = []
    testSet = []
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)
        for x in range(len(dataSet)):
            for y in range(4):
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < split:
                trainSet.append(dataSet[x])
            else:
                testSet.append(dataSet[x])
    return trainSet, testSet


def load_mnist_data(filename='./data/ClusterSamples.csv',
                    label_filename='./data/SampleLabels.csv'):
    """
    load mnist data
    :param filename:
    :param label_filename:
    :return:
    """

    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)

    with open(label_filename, 'r') as f:
        lines = csv.reader(f)
        training_data_label = list(lines)

    # transfer label to 1 dim, char to num
    training_data_label = [int(la[0]) for la in training_data_label]
    return np.array(dataSet, dtype='float64'), np.array(training_data_label)


def load_sample_data():
    """
    the sample data is to test the model if right.
    :return:
    """
    labels = None
    dataSets = [
        [0, 0], [1, 0], [0, 1], [1, 1],
        [2, 1], [1, 2], [2, 2], [3, 2],
        [6, 6], [7, 6], [8, 6], [7, 7],
        [8, 7], [9, 7], [7, 8],  [8, 8],
        [9, 8], [8, 9], [9, 9]
    ]
    return np.array(dataSets), labels


if __name__ == "__main__":
    # trainSet, testSet = load_iris_data('./data/iris.data', 0.66)
    # print(trainSet)
    # print("*"*100)
    # print(testSet)
    # load mnist data
    dataSets, labels = load_mnist_data()
    # print(dataSets)
    print(labels)


