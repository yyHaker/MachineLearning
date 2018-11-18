# -*- coding: utf-8 -*-
import numpy as np
import csv

def load_sample_data():
    """
    load sample data.
    :return:
    """
    datas = [
        [1, 1], [2, 2], [2, 0],
        [0, 0], [1, 0], [0, 1],
    ]
    labels = [1, 1, 1, 2, 2, 2]
    return np.array(datas, dtype='float64'), np.array(labels)


def load_mnist_data(filename='./data/TestSamples.csv',
                    label_filename='./data/TrueLabels.csv'):
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


if __name__ == "__main__":
    datas, labels = load_sample_data()
    print(datas)
    print(labels)
