#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Day02.py
# @Author: Funn_Y
# @Date  : 2019/2/1
# @Modify  :
# @Software : PyCharm
# ref：https://github.com/MLEveryday/100-Days-Of-ML-Code
#   ref: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day%202_Simple_Linear_Regression.md
# script_fun: 简单线性回归实现


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pylab import *


def explanfun(func):
    print("\033[1;31;0m参考: https://github.com/MLEveryday/100-Days-Of-ML-Code/blob/master/Code/Day"
          "%202_Simple_Linear_Regression.md\033[0m")
    print("ref提示\n--------------------------------")
    ref = "https://blog.csdn.net/marsjohn/article/details/54911788"
    print("最小二乘法推导: %s" % ref)
    ref = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html"
    print("sklearn.linear_model.LinearRegression: %s" % ref)
    ref = "https://matplotlib.org/api/pyplot_api.html"
    print("matplotlib.pyplot: %s" % ref)
    print("--------------------------------")
    return func


@explanfun
def mainfun():
    plt.style.use('ggplot')  # 使用'ggplot'风格美化显示的图表
    # 加载数据
    dataset = pd.read_csv('studentscores.csv')
    xx = dataset.iloc[:, :1].values
    yy = dataset.iloc[:, 1].values

    # 分割数据
    xx_train, xx_test, yy_train, yy_test = train_test_split(xx, yy, test_size=1/4, random_state=0)

    # 套用线性拟合工具
    regressor = LinearRegression(fit_intercept=True, normalize=True, copy_X=True, n_jobs=-1)
    # sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    # fit_intercept：选择是否需要计算截距，默认为True，如果中心化了的数据可以选择false
    # normalize：选择是否需要标准化（中心化），默认为false，和参数fit_intercept有关，自行思考
    # copy_X：选择是否复制X数据，默认True,如果否，可能会因为中心化把X数据覆盖
    # n_jobs:int量，选择几核用于计算，默认1，-1表示全速运行
    # 其它的模型：Ridge，Lasso,ElasticNet,ref： https://blog.csdn.net/weixin_42451864/article/details/81352878
    regressor = regressor.fit(xx_train, yy_train)

    # np.set_printoptions(precision=3, threshold=5, edgeitems=3, linewidth=2, suppress=None, nanstr=None,
    #                     infstr=None, formatter=None)
    # # precision: 设置浮点数的精度 （默认值：8）
    # # threshold: 设置显示的数目（超出部分省略号显示， np.nan是完全输出，默认值：1000）
    # # edgeitems: 设置显示前几个，后几个 （默认值：3）
    # # suppress:  设置是否科学记数法显示 （默认值：False）
    # # ref:https://blog.csdn.net/weixin_40309268/article/details/83579381

    # 数据可视化
    mpl.rcParams['font.sans-serif'] = ['SimHei']    # 为了是图片汉字显示正常
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(1)
    plt.subplot(211)    # subplot(行数，列数，第几号图)
    plt.scatter(xx_train, yy_train, color='red')
    plt.plot(xx_train, regressor.predict(xx_train), color='blue')
    plt.title("预测结果")

    plt.subplot(212)
    plt_error = regressor.predict(xx_train) - yy_train
    mean_plt_error = sum(plt_error)/len(plt_error)
    plt.plot(plt_error, 'bo', plt_error, 'r-')

    plt.plot([0, 17], [mean_plt_error, mean_plt_error], 'k--')
    plt.text(0.0, 2, r'$\mu = $' + str(mean_plt_error), fontdict={'size': '16', 'color': 'b'})
    plt.title("误差")
    plt.show()

    plt.scatter(xx_test, yy_test, color='red')
    plt.plot(xx_test, regressor.predict(xx_test), color='blue')
    plt.title("xx_test1")
    plt.show()


if __name__ == "__main__":
    mainfun()
