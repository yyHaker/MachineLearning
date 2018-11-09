#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: myutils.py
@time: 2018/10/17 14:19
"""
import pickle


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
