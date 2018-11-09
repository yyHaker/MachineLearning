#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: dataHelper.py
@time: 2018/10/16 09:01
"""
import numpy as np
import csv
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MnistDataSet(Dataset):
    """
    MNIST dataset.
    """
    def __init__(self, file_name='./data/TrainSamples.csv',
                 label_filename='./data/TrainLabels.csv',
                 split_rate=0.7,
                 data_type="train"):
        """
        :param file_name: data files.
        :param label_filename: label files.
        :param split_rate: train data split rate.
        :param data_type: "train", "valid" or "test".
        """
        self.file_name = file_name
        self.label_filename = label_filename
        self.split_rate = split_rate
        self.data_type = data_type

        if data_type == "train" or "valid":
            if self.split_rate is None:
                self.train_datas, self.train_labels = self._load_train_datas()
            else:
                self.train_datas, self.train_labels, self.valid_datas, self.valid_labels = self._load_train_datas()
        elif data_type == "test":
            self.test_datas = self._load_test_datas()
        else:
            raise Exception("data type error")

    def __getitem__(self, idx):
        """通过idx索引到数据"""
        sample = {}
        if self.data_type == "train":
            data = torch.Tensor(self.train_datas[idx])
            label = torch.LongTensor([self.train_labels[idx]])
            sample = {"data": data, "label": label}
        elif self.data_type == "valid":
            data = torch.Tensor(self.valid_datas[idx])
            label = torch.LongTensor([self.valid_labels[idx]])
            sample = {"data": data, "label": label}
        elif self.data_type == "test":
            data = torch.Tensor(self.test_datas)
            sample = {"data": data}
        else:
            raise Exception("data type error")
        return sample

    def __len__(self):
        if self.data_type == "train":
            return len(self.train_datas)
        elif self.data_type == "valid":
            return len(self.valid_datas)
        elif self.data_type == "test":
            return len(self.test_datas)
        else:
            raise Exception("data type error")

    def _load_train_datas(self):
        with open(self.file_name, 'r') as csvfile:
            lines = csv.reader(csvfile)
            dataSet = list(lines)

        with open(self.label_filename, 'r') as f:
            lines = csv.reader(f)
            training_data_label = list(lines)

        # transfer label to 1 dim, char to num
        training_data_label = [int(la[0]) for la in training_data_label]

        # to numpy array
        datas, labels = np.array(dataSet, dtype='float64'), np.array(training_data_label)

        if self.split_rate is None:
            return datas, labels
        else:
            train_datas, train_labels, valid_datas, valid_labels = \
                _split_datas(datas, labels, self.split_rate)
            return train_datas, train_labels, valid_datas, valid_labels

    def _load_test_datas(self):
        pass

    def _split_datas(self, datas, labels, split_rate):
        """
        split the datas and labels.
        :param datas:
        :param labels:
        :param split_rate:
        :param shuffle:
        :return:
        """
        assert len(datas) == len(labels), "the data length is not the same...."
        datas_labels = [(datas[i], labels[i]) for i in range(len(datas))]

        split_num = int(len(datas) * split_rate)
        train_datas_labels = datas_labels[: split_num]
        valid_datas_labels = datas_labels[split_num:]

        train_datas, train_labels = [datas_labels[0] for datas_labels in
                                     train_datas_labels], [datas_labels[1] for datas_labels in train_datas_labels]
        valid_datas, valid_labels = [datas_labels[0] for datas_labels in
                                     valid_datas_labels], [datas_labels[1] for datas_labels in valid_datas_labels]
        return np.array(train_datas, dtype='float64'), np.array(train_labels), \
               np.array(valid_datas, dtype='float64'), np.array(valid_labels)


# some other functions.

def load_mnist_data(file_name='./data/TrainSamples.csv',
                    label_filename='./data/TrainLabels.csv',
                    split_rate=0.7, shuffle=True):
    """
    load mnist data.
    ---------
    data: numpy array [81, ].
    label: int.
    :param file_name: data file name.
    :param label_filename: label file name.
    :return:
    """
    with open(file_name, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataSet = list(lines)

    with open(label_filename, 'r') as f:
        lines = csv.reader(f)
        training_data_label = list(lines)

    # transfer label to 1 dim, char to num
    training_data_label = [int(la[0]) for la in training_data_label]

    # to numpy array
    datas, labels = np.array(dataSet, dtype='float64'), np.array(training_data_label)

    if split_rate is None:
        return datas, labels
    else:
        train_datas, train_labels, valid_datas, valid_labels = \
            _split_datas(datas, labels, split_rate, shuffle=shuffle)
        return train_datas, train_labels, valid_datas, valid_labels


def _split_datas(datas, labels, split_rate, shuffle=True):
    """
    split the datas and labels.
    :param datas:
    :param labels:
    :param split_rate:
    :param shuffle:
    :return:
    """
    assert len(datas) == len(labels), "the data length is not the same...."
    datas_labels = [(datas[i], labels[i]) for i in range(len(datas))]
    if shuffle:
        np.random.shuffle(datas_labels)
    split_num = int(len(datas)*split_rate)
    train_datas_labels = datas_labels[: split_num]
    valid_datas_labels = datas_labels[split_num:]

    train_datas, train_labels = [datas_labels[0] for datas_labels in
                                 train_datas_labels], [datas_labels[1] for datas_labels in train_datas_labels]
    valid_datas, valid_labels = [datas_labels[0] for datas_labels in
                                 valid_datas_labels], [datas_labels[1] for datas_labels in valid_datas_labels]
    return np.array(train_datas, dtype='float64'), np.array(train_labels), \
           np.array(valid_datas, dtype='float64'), np.array(valid_labels)


if __name__ == "__main__":
    train_datas, train_labels, valid_datas, valid_labels = load_mnist_data()
    print("train data size: ", len(train_datas))
    print("len(train_labels: )", len(train_labels), train_labels)
    print("-"*100)
    print("valid data size: ", len(valid_datas))
    print("len(valid_lables: )", len(valid_labels), valid_labels)

    # show sample
    print("len(data)", len(train_datas[0]))
    train_image = train_datas[5].reshape(9, -1)
    plt.imshow(train_image, cmap='gray')
    plt.show()


