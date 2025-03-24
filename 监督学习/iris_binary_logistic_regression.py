"""
二分类逻辑回归实现
使用鸢尾花数据集中的两个类别进行二分类
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.font_manager as fm
from matplotlib import rcParams
from matplotlib.colors import ListedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data():
    """加载鸢尾花数据集，只选择两个类别"""
    iris = load_iris()
    # 只选择前两个类别
    X = iris.data[iris.target < 2]
    y = iris.target[iris.target < 2]
    return X, y

def plot_decision_boundary(model, X, y, title):
    """绘制决策边界"""
    # 只使用前两个特征进行可视化
    X_2d = X[:, :2]
    
    # 创建一个新的只使用前两个特征的模型
    model_2d = LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0)
    model_2d.fit(X_2d, y)
    
    # 获取模型参数
    w0 = model_2d.intercept_[0]
    w1, w2 = model_2d.coef_[0]
    
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
    
    # 绘制散点图
    colors = ['#FF6B6B', '#4ECDC4']  # 使用鲜艳的颜色
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, 
                         cmap=ListedColormap(colors),
                         alpha=0.8, 
                         edgecolors='white', s=100)
    
    # 绘制决策边界线
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # 计算并显示决策边界方程
    # 对于二分类问题，决策边界是 w0 + w1*x1 + w2*x2 = 0
    # 因此 x2 = (-w0 - w1*x1) / w2
    x1 = np.array([x_min, x_max])
    x2 = (-w0 - w1 * x1) / w2
    plt.plot(x1, x2, 'k--', label='决策边界')
    
    plt.xlabel('萼片长度')
    plt.ylabel('萼片宽度')
    plt.title(title)
    
    # 添加图例
    legend1 = plt.legend(*scatter.legend_elements(),
                        loc="upper right", 
                        title="类别")
    plt.gca().add_artist(legend1)
    
    # 显示决策边界方程
    equation = f'决策边界方程: {w0:.2f} + {w1:.2f}×萼片长度 + {w2:.2f}×萼片宽度 = 0'
    plt.text(0.02, 0.98, equation, 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.show()

def predict_new_data(model, X, sepal_length, sepal_width):
    """预测新数据的类别"""
    # 创建新数据点（使用所有4个特征）
    new_data = np.array([[sepal_length, sepal_width, 0, 0]])  # 添加其他两个特征
    
    # 标准化数据
    scaler = StandardScaler()
    scaler.fit(X)  # 使用所有特征进行拟合
    new_data_scaled = scaler.transform(new_data)
    
    # 预测
    prediction = model.predict(new_data_scaled)[0]
    
    # 计算决策函数值
    decision_value = model.decision_function(new_data_scaled)[0]
    
    print(f"\n新数据预测结果:")
    print(f"萼片长度: {sepal_length}")
    print(f"萼片宽度: {sepal_width}")
    print(f"决策函数值: {decision_value:.4f}")
    print(f"预测类别: {prediction}")
    
    return prediction

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
    plot_decision_boundary(model, X_scaled, y, '鸢尾花数据集 - 二分类逻辑回归决策边界')
    
    # 预测新数据
    predict_new_data(model, X, 5.0, 3.5)  # 示例：预测一个新的数据点

if __name__ == "__main__":
    main() 