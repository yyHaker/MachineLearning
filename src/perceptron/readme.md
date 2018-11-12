# perceptron
感知器算法是一种可以直接得到线性判别函数的线性分类方法，它采用最直观的准则，即最小错分样本数，
将错分样本到判别界面的距离之和作为准则，称为感知器准则。表示如下:
![perceptron rule](/src/perceptron/pictures/perceptron_rule.png)


## result
1. 使用单样本调整版本的感知器算法分类如下结果

![perceptron single result](/src/perceptron/pictures/perceptron_single.png)

2. 使用批量调整版本的感知器算法分类如下效果

![perceptron batch result](/src/perceptron/pictures/perceptron_batch.png)

3. 使用感知器算法解决多分类问题

    每两个训练一个分类器，总共有c(c-1)/2个分类器，测试时候如果对任意j != i，有gij(x) >= 0，则决策属于wi；其它情况，则拒识。
    
    注意：如果数据集线性不可分，感知器无法收敛！
