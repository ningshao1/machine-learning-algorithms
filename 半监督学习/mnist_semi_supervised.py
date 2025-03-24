"""
半监督学习示例：使用标签传播算法对MNIST数据集进行分类
对比纯监督学习和半监督学习的效果
"""

import numpy as np
from sklearn.datasets import load_digits  # 改用load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelSpreading  # 改用LabelSpreading
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")  # 忽略警告信息

# 设置随机种子
np.random.seed(42)

# 1. 加载数据集
print("加载数据集...")
digits = load_digits()  # 加载digits数据集
X, y = digits.data, digits.target  # 获取特征和标签

# 2. 数据预处理
# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"训练集大小: {X_train.shape[0]}，测试集大小: {X_test.shape[0]}")

# 3. 构造标记数据和未标记数据
# 减少标记数据数量到30个（每个类别3个）
n_labeled = 100
n_classes = 10
n_per_class = n_labeled // n_classes  # 每个类别3个样本

labeled_indices = []
for i in range(n_classes):
    # 找到类别 i 的样本索引
    class_indices = np.where(y_train == i)[0]
    # 随机选择样本
    selected = np.random.choice(class_indices, n_per_class, replace=False)
    labeled_indices.extend(selected)

labeled_indices = np.array(labeled_indices)
np.random.shuffle(labeled_indices)  # 打乱顺序

# 创建标记和未标记数据的标签
y_train_semi = np.full(len(y_train), -1, dtype=int)  # 初始化为 -1（未标记）
y_train_semi[labeled_indices] = y_train[labeled_indices]  # 设置标记数据的标签
print(f"标记数据数量: {np.sum(y_train_semi != -1)}，未标记数据数量: {np.sum(y_train_semi == -1)}")

# 4. 纯监督学习（只用标记数据）
print("\n=== 纯监督学习（只用标记数据） ===")
# 提取标记数据
X_labeled = X_train[labeled_indices]
y_labeled = y_train[labeled_indices]

# 训练逻辑回归模型
log_reg_supervised = LogisticRegression(random_state=42, max_iter=1000)
log_reg_supervised.fit(X_labeled, y_labeled)

# 在测试集上评估
y_pred_supervised = log_reg_supervised.predict(X_test)
accuracy_supervised = accuracy_score(y_test, y_pred_supervised)
print(f"纯监督学习测试集准确率: {accuracy_supervised:.4f}")

# 5. 半监督学习（标签传播）
print("\n=== 半监督学习（标签传播） ===")
# 使用 LabelSpreading 进行标签传播，调整参数
label_spread_model = LabelSpreading(
    kernel='knn',  # 使用KNN核函数
    n_neighbors=7,  # 设置邻居数
    alpha=0.2,  # 设置标签传播的平滑参数
    max_iter=1000,  # 增加最大迭代次数
    tol=1e-3  # 设置收敛阈值
)
label_spread_model.fit(X_train, y_train_semi)

# 获取传播后的标签
y_train_propagated = label_spread_model.predict(X_train)

# 使用传播后的标签训练逻辑回归模型
log_reg_semi = LogisticRegression(random_state=42, max_iter=1000)
log_reg_semi.fit(X_train, y_train_propagated)

# 在测试集上评估
y_pred_semi = log_reg_semi.predict(X_test)
accuracy_semi = accuracy_score(y_test, y_pred_semi)
print(f"半监督学习测试集准确率: {accuracy_semi:.4f}")

# 6. 对比
print("\n=== 对比结果 ===")
print(f"纯监督学习准确率: {accuracy_supervised:.4f}")
print(f"半监督学习准确率: {accuracy_semi:.4f}")
print(f"半监督学习提升: {(accuracy_semi - accuracy_supervised):.4f}") 