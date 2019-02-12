#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Day03.py
# @Author: Funn_Y
# @Date  : 2019/2/2
# @Modify  :
# @Software : PyCharm
# ref: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%203_Multiple_Linear_Regression.md
# script_fun: 多元线性回归


import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pylab import *
from matplotlib import font_manager


my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
# 加载数据
dataset = pd.read_csv('50_Startups.csv')
xx = dataset.iloc[:, :-1].values
yy = dataset.iloc[:, 4].values

# lable并创建虚拟变量
labelencoder = LabelEncoder()
xx[:, 3] = labelencoder.fit_transform(xx[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[xx.shape[1] - 1])
xx = onehotencoder.fit_transform(xx).toarray()
# np.savetxt("xx.txt", xx)

xx = xx[:, 1:]  # 为什么舍弃第0列？

xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(xx_train, yy_train)
y_pred = regressor.predict(xx_test)

plt.style.use('ggplot')  # 使用'ggplot'风格美化显示的图表
plt.figure(1)
plt.subplot(211)  # subplot(行数，列数，第几号图)
plt.plot(y_pred, 'r', yy_test, 'k')
plt.title("预测结果", fontproperties=my_font)
plt.plot(y_pred, label=u"预测结果")
plt.plot(yy_test, label=u"真实结果")
plt.legend(prop=my_font)

plt.subplot(212)
plt_error = y_pred - yy_test
mean_plt_error = sum(plt_error) / len(plt_error)
plt.plot(plt_error, 'bo', plt_error, 'r-')
plt.plot([0, 10], [mean_plt_error, mean_plt_error], 'k--')

plt.show()
