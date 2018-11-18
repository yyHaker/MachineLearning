#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: myutils.py
@time: 2018/10/17 14:19
"""
import pickle
import csv
import numpy as np


def load_data_from_file(path):
    """
    :param path: the store path
    :return:
    """
    data_obj = None
    with open(path, 'rb') as f:
        data_obj = pickle.load(f)
    return data_obj


def write_data_to_file(data, path):
    """
    :param data: the data obj
    :param path: the store path
    :return:
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def write_to_csv(datas, path):
    """
    :param datas: a list of data.
    :param path: path
    :return:
    """
    datas = np.array(datas)
    datas = np.reshape(datas, (len(datas), 1))
    # 用csv一行一行的写入,
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(datas)


def read_from_csv(path):
    """
    :param path: path.
    :return: lines.
    """
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        lines = list(reader)
    return lines


def calc_acc(csv1, csv2):
    """
    compare two files, calc acc.
    :param csv1: path
    :param csv2: path
    :return:
    """
    count = 0.
    lines1 = read_from_csv(csv1)
    lines2 = read_from_csv(csv2)
    for i, j in zip(lines1, lines2):
        if int(i[0]) == int(j[0]):
            count += 1
    return count/len(lines1)

