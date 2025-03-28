# 机器学习算法练习

这个仓库包含各种机器学习算法的实现和练习。

## 目录

- 监督学习
  - 线性回归
  - 逻辑回归
  - 决策树
  - 随机森林
  - 支持向量机
  - 朴素贝叶斯

- 无监督学习
  - K均值聚类
  - 层次聚类
  - 主成分分析
  - 关联规则学习

- 深度学习
  - 神经网络基础
  - 卷积神经网络
  - 循环神经网络
  - 生成对抗网络

- 半监督学习
  - MNIST数据集标签传播

## 使用说明

每个算法文件夹包含：
1. 算法实现代码
2. 示例数据集
3. 练习与解答
4. 算法原理说明

## 环境要求

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/PyTorch（深度学习部分）
- Matplotlib

## 半监督学习

### MNIST数据集标签传播
- 文件：`半监督学习/mnist_semi_supervised.py`
- 描述：使用标签传播算法对MNIST数据集进行半监督学习
- 特点：
  - 使用少量标记数据（每个类别10个样本）
  - 对比纯监督学习和半监督学习的效果
  - 使用标签传播算法进行标签预测
  - 使用逻辑回归作为最终分类器

## 注意事项

- 运行MNIST数据集示例时，首次运行会自动下载数据集，可能需要一些时间
- 建议使用Python 3.7或更高版本
- 确保有足够的内存运行MNIST数据集示例