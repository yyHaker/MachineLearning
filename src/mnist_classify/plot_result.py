#!/usr/bin/python
# coding:utf-8

"""
@author: yyhaker
@contact: 572176750@qq.com
@file: plot_result.py
@time: 2018/10/17 14:18
"""
# -*- coding:utf-8 -*-
"""
plot the result.
losses_dict = {"train_loss": [], "valid_loss": []}
    acc_dic = {"train_acc": [], "valid_acc": []}
"""
import numpy as np
import matplotlib.pyplot as plt
from myutils import load_data_from_file

losses_dict = load_data_from_file("result/loss_dict.pkl")
acc_dict = load_data_from_file("result/acc_dict.pkl")

train_loss, valid_loss = losses_dict["train_loss"], losses_dict["valid_loss"]
train_acc, valid_acc = acc_dict['train_acc'], acc_dict['valid_acc']
print("len(train_loss): ", len(train_loss))
assert len(train_loss) == len(valid_loss) and len(train_acc) == len(valid_acc)
# plot data
epoches = np.arange(1, len(train_loss)+1)

plt.figure()
plt.plot(epoches, train_loss, label="train_loss")
plt.plot(epoches, valid_loss, color='red', linewidth=1.0, linestyle='--', label="valid_loss")
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("loss")
plt.title("train and valid loss")
plt.show()

plt.figure()
plt.plot(epoches, train_acc, label="train_acc")
plt.plot(epoches, valid_acc, label="valid_acc")
plt.legend()
plt.xlabel("EPOCH")
plt.ylabel("accuracy")
plt.title("train and valid accuracy")
plt.show()
