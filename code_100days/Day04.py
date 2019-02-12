#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Day_04.py
# @Author: Funn_Y
# @Date  : 2019/2/12
# @Modify  :
# @Software : PyCharm
# ref: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%206_Logistic_Regression.md
# script_fun:逻辑回归


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap


def explanfun(func):
    print("\033[1;31;0m参考: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day"
          "%206_Logistic_Regression.md\033[0m")
    print("ref提示\n--------------------------------")
    ref = "https://www.cnblogs.com/weiququ/p/8085964.html"
    print("逻辑回归的理解: %s" % ref)
    ref = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
    print("sklearn.linear_model.LogisticRegression: %s" % ref)
    ref = "https://blog.csdn.net/qq_38683692/article/details/82533460"
    print("中文版: %s" % ref)
    ref = "https://www.cnblogs.com/pinard/p/6035872.html"
    print("中文版 参数选择: %s" % ref)
    ref = "https://blog.csdn.net/langb2014/article/details/51118792"
    print("混淆矩阵: %s" % ref)
    print("--------------------------------")
    return func


@explanfun
def mainfun():
    dataset = pd.read_csv('Social_Network_Ads.csv')
    xx = dataset.iloc[:, [2, 3]].values
    xx = np.array(xx, dtype='float64')
    yy = dataset.iloc[:, 4].values
    yy = np.array(yy, dtype='float64')

    xx_train, xx_test, y_train, y_test = train_test_split(xx, yy, test_size=0.25, random_state=0)

    sc = StandardScaler()
    xx_train = sc.fit_transform(xx_train)
    xx_test = sc.transform(xx_test)

    # LogisticRegression用法
    """"
    LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
    class_weight=None, random_state=None, solver='warn', max_iter=100, multi_class='warn', verbose=0, warm_start=False, 
    n_jobs=None)[source]
    参数选择：https://www.cnblogs.com/pinard/p/6035872.html
    penalty参数可选择的值为"l1"和"l2".分别对应L1的正则化和L2的正则化，默认是L2的正则化。
    solver参数
        a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
        b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
        d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，
        适合于样本数据多的时候，SAG是一种线性收敛算法，这个速度远比SGD快。
        newton-cg, lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有连续导数的L1正则化，
        只能用于L2正则化。而liblinear通吃L1正则化和L2正则化。
        MvM一般比OvR分类相对准确一些，但是liblinear只支持OvR，不支持MvM。如果我们需要相对精确的多元逻辑回归时，就不能
        选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了
    """
    classifier = LogisticRegression(solver='liblinear')
    classifier.fit(xx_train, y_train)

    # 生成混淆矩阵
    # 原代码中没有用到相关计算结果
    # 混淆矩阵是未来说明模型预测结果如何
    # 实际真个数为P,实际假的个数为Q
    # 真实结果          预测结果        数量
    #   真       -->     真               a
    #   假       -->     真               b
    #   真       -->     假               c
    #   假       -->     假               d
    # 则模型整体准确率：(a+d)/(a+b+c+d)
    # 模型对真的准确率：a/P
    # 模型对假的准确率：d/Q
    y_pred = classifier.predict(xx_test)
    cm = confusion_matrix(y_test, y_pred)
    labels_name = ["true", "false"]
    mytitle = "Confusion Matrix"
    plot_confusion_matrix(cm, labels_name, mytitle)

    plt.figure()
    xx_set, y_set = xx_train, y_train
    xx1, xx2 = np.meshgrid(np. arange(start=xx_set[:, 0].min()-1, stop=xx_set[:, 0].max()+1, step=0.01),
                           np.arange(start=xx_set[:, 1].min()-1, stop=xx_set[:, 1].max()+1, step=0.01))

    plt.contourf(xx1, xx2, classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for i, j in enumerate(np. unique(y_set)):
        plt.scatter(xx_set[y_set == j, 0], xx_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt.title(' LOGISTIC(Training set)')
    plt.xlabel(' Age')
    plt.ylabel(' Estimated Salary')
    plt.legend()
    plt.show()

    plt.figure()
    xx_set, y_set = xx_test, y_test
    xx1, xx2 = np.meshgrid(np. arange(start=xx_set[:, 0].min()-1, stop=xx_set[:, 0].max()+1, step=0.01),
                           np.arange(start=xx_set[:, 1].min()-1, stop=xx_set[:, 1].max()+1, step=0.01))

    plt.contourf(xx1, xx2, classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # print(y_set)
    # print("***********")
    # print(xx_set)
    for i, j in enumerate(np.unique(y_set)):
        # print("(i=%s,j=%s)" % (i, j))
        plt.scatter(xx_set[y_set == j, 0], xx_set[y_set == j, 1],
                    c=ListedColormap(('red', 'green'))(i), label=j)

    plt. title(' LOGISTIC(Test set)')
    plt. xlabel(' Age')
    plt. ylabel(' Estimated Salary')
    plt. legend()
    plt. show()


def plot_confusion_matrix(cm, labels_name, mytitle):
    plt.figure()
    # plot_confusion_matrix(cm, labels_name, "HAR Confusion Matrix")
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(mytitle)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    mainfun()
