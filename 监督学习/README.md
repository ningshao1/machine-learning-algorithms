# 监督学习算法实现

本目录包含各种监督学习算法的实现和示例。

## 目录

- [鸢尾花逻辑回归分类](./iris_logistic_regression.py) - 使用逻辑回归算法实现鸢尾花数据集的多分类任务

## 鸢尾花逻辑回归分类

这是一个经典的机器学习入门案例，专注于使用逻辑回归算法进行多分类。本实现包括：

- 详细的数据分析和可视化
- 逻辑回归模型训练与评估
- 决策边界可视化
- 模型系数分析
- 学习曲线分析
- 正则化参数调优

### 运行方法

```bash
# 确保已安装所需依赖
pip install numpy pandas matplotlib scikit-learn

# 运行鸢尾花逻辑回归分类程序
python iris_logistic_regression.py
```

### 输出内容

程序将输出：
1. 数据集基本信息
2. 逻辑回归模型在测试集上的准确率
3. 交叉验证结果
4. 分类报告（精确率、召回率、F1值）
5. 混淆矩阵
6. 模型系数分析
7. 不同正则化参数的性能对比

同时会生成多个可视化图表：
- `iris_visualization.png` - 特征对散点图
- `iris_boxplots.png` - 特征分布箱型图
- `confusion_matrix.png` - 混淆矩阵可视化
- `decision_boundary.png` - 逻辑回归决策边界
- `feature_coefficients.png` - 特征系数分析
- `learning_curve.png` - 学习曲线
- `regularization_impact.png` - 正则化参数影响

## 监督学习

### 1. 逻辑回归 (Logistic Regression)

#### 1.1 多分类逻辑回归 (iris_logistic_regression.py)
- 使用sklearn内置的鸢尾花数据集
- 实现了多分类逻辑回归
- 包含数据预处理、模型训练、评估和可视化
- 使用StandardScaler进行特征标准化
- 使用交叉验证评估模型性能
- 可视化决策边界

#### 1.2 二分类逻辑回归 (iris_binary_logistic_regression.py)
- 使用鸢尾花数据集中的两个类别进行二分类
- 实现了二分类逻辑回归
- 包含数据预处理、模型训练、评估和可视化
- 使用StandardScaler进行特征标准化
- 使用交叉验证评估模型性能
- 可视化决策边界和ROC曲线
- 计算并展示混淆矩阵
- 输出精确率、召回率和F1分数

### 2. 支持向量机 (SVM)
- 待实现

### 3. 决策树 (Decision Tree)
- 待实现

### 4. 随机森林 (Random Forest)
- 待实现

### 5. K近邻 (K-Nearest Neighbors)
- 待实现

### 6. 朴素贝叶斯 (Naive Bayes)
- 待实现

### 7. 神经网络 (Neural Network)
- 待实现

## 无监督学习
- 待实现

## 强化学习
- 待实现 