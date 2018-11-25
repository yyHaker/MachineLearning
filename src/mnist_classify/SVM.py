#!/usr/bin/python
# coding:utf-8

"""
使用SVM来对MNIST数据进行分类
@author: yyhaker
@contact: 572176750@qq.com
@file: SVM.py
@time: 2018/11/25 16:09
"""
import numpy as np
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from dataHelper import load_mnist_data


def main():
    # load train data and test data
    print("loading data...")
    train_datas, train_labels = load_mnist_data(file_name='data/train_data/TrainSamples',
                                                label_filename='data/train_data/TrainLabels')
    test_datas, test_labels = load_mnist_data(file_name='data/valid_data/ValidSamples',
                                              label_filename='data/valid_data/ValidLabels')
    print("Train data size: {}, Test data size: {}".format(len(train_datas), len(test_datas)))

    # 对训练和测试的特征数据进行标准化
    # ss = StandardScaler()
    # train_datas = ss.fit_transform(train_datas)
    # test_datas = ss.transform(test_datas)

    # create models and train
    print("training the model....")
    clf = OneVsRestClassifier(estimator=svm.SVC(C=10, kernel='rbf', gamma='auto'), n_jobs=4)
    clf.fit(train_datas, train_labels)
    print("training is done!")
    print("fit result: ", clf.score(train_datas, train_labels))
    # test on the test data
    acc = clf.score(test_datas, test_labels)
    print("test acc: ", acc)
    predict = clf.predict(test_datas)
    print("Classification report:\n ", metrics.classification_report(test_labels, predict))
    print("Confusion matrix:\n ", metrics.confusion_matrix(test_labels, predict))


if __name__ == "__main__":
    main()

