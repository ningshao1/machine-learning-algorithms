#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
鸢尾花数据集逻辑回归多分类实现
===============================
使用逻辑回归算法对鸢尾花数据集进行多分类预测

数据集描述:
- 特征: 4个数值特征 (萼片长度, 萼片宽度, 花瓣长度, 花瓣宽度)
- 目标: 3个类别 (Setosa, Versicolour, Virginica)
- 样本数: 150个样本 (每个类别50个)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import json
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# # 设置随机种子，保证结果可复现
# np.random.seed(42)


def load_data():
    """加载鸢尾花数据集"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    return X, y

def plot_decision_boundary(model, X, y, title):
    """绘制决策边界"""
    # 只使用前两个特征进行可视化
    X_2d = X[:, :2]
    
    # 创建一个新的只使用前两个特征的模型
    model_2d = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
    model_2d.fit(X_2d, y)
    
    # 创建网格点
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    
    # 设置背景颜色映射（使用浅色系）
    plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.Pastel1)
    
    # 设置散点颜色（使用深色系）
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # 使用鲜艳的颜色
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                         cmap=ListedColormap(colors),
                         alpha=0.8, 
                         edgecolors='white', s=100)
    
    plt.xlabel('萼片长度')
    plt.ylabel('萼片宽度')
    plt.title(title)
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", 
                        title="类别")
    plt.gca().add_artist(legend1)
    
    plt.show()

def main():
    # 加载数据
    X, y = load_data()
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 创建并训练模型
    model = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 输出模型评估结果
    print("\n模型评估结果:")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    print("\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    # 交叉验证
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print("\n交叉验证结果:")
    print(f"平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # 绘制决策边界
    plot_decision_boundary(model, X_scaled, y, '鸢尾花数据集 - 逻辑回归决策边界')

if __name__ == "__main__":
    main() 