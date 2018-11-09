# -*- coding: utf-8 -*-
from data_helper import load_mnist_data
from sklearn.cluster import KMeans
import numpy as np


def evaluate_cluster(labels, predict):
    """
    calc the acc of cluster.
    :param labels: array like
    :param predict: array like
    :return:
     'accuracy', float
    """
    return np.sum(labels == predict) / labels.shape[0]


if __name__ == "__main__":
    train_data, train_data_label = load_mnist_data()
    # k-means
    kmeans = KMeans(n_clusters=10, init='random')
    print("begin clustering.....")
    kmeans = kmeans.fit(train_data)
    print("end clustering")
    # get the labels
    predict_labels = kmeans.predict(train_data)

    # calc acc
    print("train_data_label: ", train_data_label)
    print("predict labels: ", predict_labels)
    print("acc: ", evaluate_cluster(train_data_label, predict_labels))
