#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Day01.py
# @Author: Funn_Y
# @Date  : 2019/2/1
# @Modify  :
# @Software : PyCharm
# ref：https://github.com/MLEveryday/100-Days-Of-ML-Code
#   ref: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%201_Data_Preprocessing.md
# script_fun: 100-Days-Of-ML-Code  -- 数据与处理


import numpy as np
import pandas as pd
# from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

"""
Data.csv
Country	Age	Salary	Purchased
France	44	72000	No
Spain	27	48000	Yes
Germany	30	54000	No
Spain	38	61000	No
Germany	40		    Yes
France	35	58000	Yes
Spain		52000	No
France	48	79000	Yes
Germany	50	83000	No
France	37	67000	Yes
"""


def explanfun(func):
    print("\033[1;31;0mref:https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day"
          "%201_Data_Preprocessing.md\033[0m")
    print("ref提示\n--------------------------------")
    ref = "https://blog.csdn.net/w_weiying/article/details/81411257"
    print("DataFrame.iloc: %s" % ref)
    ref = "https://blog.csdn.net/kancy110/article/details/75041923"
    print("\033[1;31;0m弃用sklearn.preprocessing.Imputer: %s\033[0m" % ref)
    ref = "https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html"
    print("sklearn.impute.SimpleImputer: %s" % ref)
    ref = "https://blog.csdn.net/accumulate_zhang/article/details/78510571\n" \
          "     旧版说明 https://www.cnblogs.com/zhoukui/p/9159909.html"
    print("sklearn.preprocessing.OneHotEncoder: %s" % ref)
    ref = "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html\n" \
          "     中文版说明 https://blog.csdn.net/u012609509/article/details/78554709"
    print("sklearn.model_selection.train_test_split: %s" % ref)
    print("--------------------------------")
    return func


@explanfun
def mainfun():
    # 数据加载
    dataset = pd.read_csv("Data.csv")  # dataset为一个DataFrame结构
    xx = dataset.iloc[:, :-1].values    # iloc与loc区别，一个按列号(从0开始)一个按列名
    yy = dataset.iloc[:, 3].values
    # print(xx)
    # print(yy)

    # 数据清洗(针对原数据，主要是缺失数据的补充)
    imp_mean = SimpleImputer(missing_values=np.nan)
    # 创建一个替换的类，其替换原则是全列的均值来代替缺陷值
    # 原文中Imputer被替换成SimpleImputer,加载包也有相应的更新
    # from sklearn.preprocessing import Imputer
    # from sklearn.impute import SimpleImputer
    imp_mean = imp_mean.fit(xx[:, 1:3])
    xx[:, 1:3] = imp_mean.transform(xx[:, 1:3])

    # 数据转换
    labelencoder_xx = LabelEncoder()  # 创建一个类，用于将文字转为数字量
    xx[:, 0] = labelencoder_xx.fit_transform(xx[:, 0])
    labelencoder_yy = LabelEncoder()
    yy = labelencoder_yy.fit_transform(yy)
    # print(xx)

    # 创建虚拟变量
    onehotencoder = OneHotEncoder(categories='auto')    # 原文这里使用的是老版本
    """
    这个函数比较难理解，举个例子
    # 一个4*3维list
    [
       [0, 1, 2],
       [1, 0, 3],
       [0, 2, 0],
       [0, 1, 1]
    ]
    第一列有2个纬度：0，1
    第二列有3个纬度：0，1，2
    第三列有4个纬度：0，1，2，3
    OneHotEncoder后第一行解析结果为
    0 -->10(因为第一列就2个纬度0，1；10表示低1个纬度为真，即为0)
    1 -->010(因为第二列就3个纬度0，1,2；010表示低1个纬度为真，即为1)
    2 -->0010(因为第三列就4个纬度0，1,2，3；010表示低1个纬度为真，即为2)
    类似的，如果某列有10个纬度，而某个数为8，则其解析结果应该为 0000000100
    """
    xx = onehotencoder.fit_transform(xx).toarray()
    # 如果不加 toarray() 的话，输出的是稀疏的存储格式，toarray()是为了方便使用
    # print(xx)

    # 部分数据用于训练，部分数据用于测试；
    xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=0)
    # test_size：可以为浮点、整数或None，默认为None(对应的为0.25)
    # 若为浮点时，表示测试集占总样本的百分比
    # 若为整数时，表示测试样本样本数
    # random_state：可以为整数、RandomState实例或None，默认为None
    # 若为None时，每次生成的数据都是随机，可能不一样
    # 若为整数时，每次生成的数据都相同

    # 数据特征量化:StandardScaler是取均值和标准差来归一
    # 即：[Xi-mean(X)]/std(X) X为某一列向量
    # print("xx_train\n", xx_train)
    sc_x = StandardScaler()
    xx_train = sc_x.fit_transform(xx_train)
    xx_test = sc_x.transform(xx_test)
    # print("xx_train\n", xx_train)
    # print("xx_test\n", xx_test)


if __name__ == "__main__":
    print("主要是通过sklearn进行数据分析，这里使用了数据ETL(对于缺陷数据的补充，利用均值，SimpleImputer)")
    print("对数据标签的转化，这里使用了LabelEncoder & OneHotEncoder")
    print("对数据的分割，取部分数据为测试数据，部分数据为训练数据，这里使用了train_test_split")
    print("对数据的特征量化，采用[Xi-mean(X)]/std(X)方案，这里使用了StandardScaler")
    mainfun()
