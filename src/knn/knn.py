# -*- coding: utf-8 -*-
from __future__ import absolute_import
import math
import operator
from data_helper import load_iris_data


class KnnClassifier(object):
    """K-nearest neighbor classifier algorithm."""
    def __init__(self, training_data, k):
        """
        :param training_data: training data, the last dim is label.
        :param k: k value.
        """
        self.training_data = training_data
        self.k = k

    def predict(self, sample):
        pass

    def test(self, test_data):
        """
        use the test data to test the model, return accuracy.
        :param test_data: a list of list data, the last dim is label.
        :return:
        """
        predictions = []
        for i in range(len(testSet)):
            neighbors = self.getNeighbors(self.training_data, testSet[i], self.k)
            response = self.getResponse(neighbors)
            predictions.append(response)
        accuracy = self.getAccuracy(test_data, predictions)
        return accuracy

    def getAccuracy(self, testSet, predictions):
        """
        calc accuracy.
        :param testSet:
        :param predictions:
        :return:
        """
        correct = 0
        for i in range(len(testSet)):
            if testSet[i][-1] == predictions[i]:
                correct += 1
        return (correct / float(len(testSet))) * 100

    def getResponse(self, neighbors):
        """
        基于最近的实例来预测结果.
        :param neighbors:
        :return:
        """
        classVotes = {}
        for i in range(len(neighbors)):
            response = neighbors[i][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1),
                             reverse=True)
        return sortedVotes[0][0]

    def getNeighbors(self, training_set, testInstance, k):
        """
        get k  nearest neighbors of one sample.
        :param training_set: train data.
        :param testInstance: test data.
        :param k: k
        :return:
        """
        distances = []
        length = len(testInstance) - 1
        for data in training_set:
            dist = self.euclideanDistance(data, testInstance, length)
            distances.append((data, dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        # k neighbors
        for i in range(k):
            neighbors.append(distances[i][0])
        return neighbors

    def euclideanDistance(self, instance1, instance2, length):
        """
        calc euclidean distance.
        :param instance1:
        :param instance2:
        :param length: dim
        :return:
        """
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


if __name__ == "__main__":
    print("load iris data...")
    trainingSet, testSet = load_iris_data('./data/iris.data', 0.66)
    print("load data done, training set size {}, test set size {}".format(len(trainingSet), len(testSet)))
    knn_classifier = KnnClassifier(trainingSet, k=3)
    acc = knn_classifier.test(testSet)
    print("accuracy: ", acc)
