# -*- coding: utf-8 -*-
import numpy as np


def load_sample_data():
    """
    load sample data.
    :return:
    """
    datas = [
        [0, 0], [1, 0], [0, 1],
        [1, 1], [2, 2], [2, 0],
    ]
    labels = [0, 0, 0, 1, 1, 1]
    return np.array(datas, dtype='float64'), np.array(labels)


def load_sample1_data():
    """
    load sample1 data.
    :return:
    """
    datas = [
        [0, 0], [0, 1],
        [1, 0], [1, 1]
    ]
    labels = [0, 0, 1, 1]
    return np.array(datas, dtype='float64'), np.array(labels)


if __name__ == "__main__":
    datas, labels = load_sample_data()
    print(datas)
    print(labels)
