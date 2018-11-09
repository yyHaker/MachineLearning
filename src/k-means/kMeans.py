# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib.pyplot as plt
from data_helper import load_mnist_data, load_sample_data


class KnnCluster(object):
    """k-means cluster."""
    def __init__(self, dataSet, K, labels=None):
        """
        :param dataSet: train data.
        :param K: the number of clusters.
        :param labels: train data labels.
        """
        self.dataSet = dataSet
        self.K = K
        self.labels = labels

    def clusters(self):
        """
        clusters .
        聚类直到质心基本确定.
        :return:
        """
        # init centroids
        self.centroidsList = self._init_centroids()

        clusterDict, clusterCount = self._cluster()
        print("current cluster dict count: ", clusterCount)
        newVar = self.getVar(self.centroidsList, clusterDict)

        oldVar = 1
        while abs(oldVar - newVar) >= 0.0001:
            self.centroidsList = self.getCentroids(clusterDict)
            clusterDict, clusterCount = self._cluster()
            print("current cluster dict count: ", clusterCount)

            oldVar = newVar
            newVar = self.getVar(self.centroidsList, clusterDict)
            print("var diff: ", abs(oldVar-newVar))
        return self.centroidsList, clusterDict, clusterCount

    def _cluster(self):
        """
        cluster for one time.
        对于每个dataSet的item，计算item与centroidList中K个质心的距离，
        找出最小的距离，并将item加入相应的聚类中.
        :return:
          'clusterDict: {class1: [data1..], class2: [data2...], ...}'
        """
        # dict 保存聚类结果
        clusterDict = {}

        # dict 保存类别数目
        cluster_count = {}

        # 将数据加到距离最近的cluster中
        for idx in range(self.dataSet.shape[0]):
            vec1 = self.dataSet[idx]
            minDis = float("inf")
            flag = -1  # 保存类别标记
            for i in range(self.K):
                vec2 = self.centroidsList[i]
                distance = self._calcDistance(vec1, vec2)
                if distance < minDis:
                    minDis = distance
                    flag = i
            if flag not in clusterDict.keys():
                clusterDict[flag] = []
            clusterDict[flag].append(vec1)

            # 记录这个cluster中的类别数目
            if self.labels is not None:
                if flag not in cluster_count.keys():
                    cluster_count[flag] = {}
                label = int(self.labels[idx])
                if label not in cluster_count[flag].keys():
                    cluster_count[flag][label] = 1
                else:
                    cluster_count[flag][label] += 1
        return clusterDict, cluster_count

    def getCentroids(self, clusterDict):
        """
         重新计算k个质心.
        :param clusterDict:
        :return:
        """
        centroidList = []
        for key in clusterDict.keys():
            centroids = np.mean(clusterDict[key], axis=0)
            centroidList.append(centroids)
        return centroidList

    def getVar(self, centroidList, clusterDict):
        """
        计算各族集合间的均方误差.
        :param centroidList:
        :param clusterDict:
        :return:
        """
        sum = 0.0
        for key in clusterDict.keys():
            vec1 = centroidList[key]
            distance = 0.0
            for item in clusterDict[key]:
                vec2 = item
                distance += self._calcDistance(vec1, vec2)
            sum += distance / len(clusterDict[key])
        return sum / self.K

    def _init_centroids(self):
        """
        随机选择k个中心点
        :return:
        """
        dataSet = list(self.dataSet)
        return random.sample(dataSet, self.K)

    def _calcDistance(self, vec1, vec2):
        """
        计算欧式距离.
        :param vec1:
        :param vec2:
        :return:
        """
        return np.linalg.norm(vec1 - vec2)

    def showCluster(self, centroidList, clusterDict):
        """
         展示聚类结果.(仅针对二维度数据)
        :param centroidList:
        :param clusterDict:
        :return:
        """
        print("centroidList: ", centroidsList)
        colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'ow']  # 不同簇类标记，o表示圆形，另一个表示颜色
        centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dw']

        # plot the cluster result
        for key in clusterDict.keys():
            plt.plot(centroidList[key][0], centroidList[key][1], centroidMark[key], markersize=12)  # 质心点
            for item in clusterDict[key]:
                plt.plot(item[0], item[1], colorMark[key])
        plt.show()


if __name__ == "__main__":
    dataSets, labels = load_mnist_data()
    # dataSets, labels = load_sample_data()
    knn_cluster = KnnCluster(dataSets, K=10, labels=labels)
    centroidsList, clusterDict, clusterCount = knn_cluster.clusters()
    # print(centroidsList)
    # print(clusterDict)
    print("cluster dict count: ")
    for k, v in clusterDict.items():
        print("class {}: {}".format(k, len(v)))
    print("final clustering result count(detail): ")
    print(clusterCount)
    # knn_cluster.showCluster(centroidsList, clusterDict)
