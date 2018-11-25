# MNIST手写数字分类任务
使用机器学习的方法来分类手写数字识别，该数据集总共有20000个，包含10个类别。
每个样本数据是压缩过后的数据，有81维，用灰度图像看看不清楚！

# 数据集划分
Total data size|Train data size|Valid data size
------------|-----------|----------------
20000       |   16000    |    4000

# 模型与结果

* 方法一: K-NN

models | train accuracy| test accuracy
--------|---------------|-----------------
KNN+std|0.943 | 0.9065
KNN| 0.9757    | 0.958 

* 方法二：朴素贝叶斯NB

models | train accuracy| test accuracy
--------|---------------|-----------------
NB| 0.881   | 0.873
NB+std|0.881  |   0.873

* 方法三：支持向量机SVM

models | train accuracy| test accuracy
--------|---------------|-----------------
SVM|    1.0 | 0.113
SVM+std| 1.0 |   0.966


* 方法四：神经网络(DNN)

models | train accuracy| valid accuracy| test accuracy|iteration epochs
--------|---------------|-----------------|---------------|-----------------
DNN[100, 10]| 0.995    | 0.9317  |  | 15
DNN[500,300,10]+BN |0.9971   | 0.9692       |      |  150
DNN[500,300,10]+BN+drop_out[0.1]| 0.9847  | 0.9655 | | 150  
DNN[500,300,10]+BN+drop_out[0.3]| 0.9702   | 0.9617  |  |200
DNN[500,300,100,10]+BN+drop_out[0.3]| 0.9748   | 0.9702  |  |300
DNN[500,300,200,10]+BN+drop_out[0.3]| 0.9767   | 0.9710  |  |300
DNN[500,300,300,10]+BN+drop_out[0.3]| 0.9765  | 0.9691  |  |300
DNN[600,300,200,10]+BN+drop_out[0.3]| 0.9799  | 0.9711  |  |300
DNN[800,300,200,10]+BN+drop_out[0.3]| 0.9830  | 0.9732  |  |300
DNN[800,500,200,10]+BN+drop_out[0.3]| 0.99234  | 0.9755  |  |350



## 最好的结果
训练正确率变化
![acc](/src/mnist_classify/result/acc.png)


训练loss变化
![loss](/src/mnist_classify/result/loss.png)

