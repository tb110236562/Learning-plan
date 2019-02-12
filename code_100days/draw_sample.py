#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : draw_sample.py
# @Author: Funn_Y
# @Date  : 2019/2/2
# @Modify  :
# @Software : PyCharm
# ref: https://www.cnblogs.com/zhizhan/p/5615947.html
# ref: https://blog.csdn.net/qq_42467563/article/details/82779147
# ref: http://blog.sina.com.cn/s/blog_b3a4f3f80101gq2i.html
# script_fun: 画图样例

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


X = np.linspace(-np.pi, np.pi, 100)
plt.figure(figsize=(6, 5))
Y_x2 = np.cos(X)
Y_x3 = np.sin(X)

# 先确定字体，以免无法识别汉字
my_font = font_manager.FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.subplot(111)  # 第一个参数表示：行，第二个参数表示；列，第三个参数；当前图例中的激活位置
plt.xlabel(u'X数值', fontproperties=my_font)
plt.ylabel(u'Y数值', fontproperties=my_font)
plt.title(u"函数图像", fontproperties=my_font, fontsize=16)     # 字体大小放在后面,常规设置字体都有默认大小
plt.plot(X, Y_x2, label=u"$X^{2}$函数")
plt.plot(X, Y_x3, label=u"sin(X)函数")
plt.legend(prop=my_font)
plt.style.use('ggplot')  # 使用'ggplot'风格美化显示的图表

# 设置X,Y轴的上下限
plt.xlim(-np.pi, np.pi)
plt.ylim(-1, 1)

# 设置关键刻度
plt.xticks([-np.pi, -np.pi / 2.0, np.pi / 2, np.pi])

# 添加标注。xy：标注箭头想要指示的点，xytext:描述信息的坐标
plt.annotate('note!!', xy=(-np.pi / 2, -1), xytext=(-np.pi / 2, -0.25), fontsize=16,
             arrowprops=dict(facecolor='r', shrink=0.01))
plt.scatter(-1.571, -1, s=100, c='r', marker='o')

# 添加文字,第一个参数是x轴坐标，第二个参数是y轴坐标，以数据的刻度为基准
plt.plot([-3.142, 3.142], [0, 0], 'k--')
plt.text(-3, 0.05, "0值线", fontproperties='SimHei', fontsize=26, color='k')

# 添加文字，加入转义字符
plt.text(0.4, -0.9, r'$(\mu_{mol} * m^{-2} *s^{-1})$', fontsize=15,
         horizontalalignment='center', verticalalignment='center')

plt.show()
