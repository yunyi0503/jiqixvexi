# Fashion MNIST 分类任务 - 多层感知机与支持向量机对比

本项目使用 Fashion MNIST 数据集进行图像分类，并对比多层感知机（MLP）和支持向量机（SVM）两种机器学习模型的性能表现。

## 项目概述

Fashion MNIST 是一个包含 70,000 张 28x28 像素的灰度图像的数据集，表示 10 类不同的服装物品。本项目的主要目标是:
- 使用多层感知机（MLP）和支持向量机（SVM）两种不同的机器学习模型进行图像分类
- 评估和比较两种模型的性能
- 可视化分类结果和模型表现

## 数据集详情

- **尺寸**: 70,000 张 28x28 像素灰度图像
- **分类**: 训练集 (60,000 图像) 和测试集 (10,000 图像)
- **类别**: 10 个类别的服装物品
  - 0: T-shirt/top (T恤/上衣)
  - 1: Trouser (裤子)
  - 2: Pullover (套头衫)
  - 3: Dress (连衣裙)
  - 4: Coat (外套)
  - 5: Sandal (凉鞋)
  - 6: Shirt (衬衫)
  - 7: Sneaker (运动鞋)
  - 8: Bag (包)
  - 9: Ankle boot (踝靴)

## 项目结构

```
.
├── data/                          # 数据目录
│   ├── fashion-mnist_train.csv    # 训练集数据
│   └── fashion-mnist_test.csv     # 测试集数据
├── models/                        # 保存的模型文件
│   ├── svm_linear_C1_gammascale.pkl  # 线性核SVM模型
│   ├── svm_poly_C1_gammascale_d3.pkl # 多项式核SVM模型
│   ├── svm_poly_C10_gammascale_d3.pkl# 多项式核SVM模型(不同参数)
│   ├── svm_rbf_C0.1_gammascale.pkl   # RBF核SVM模型
│   ├── svm_rbf_C1_gammascale.pkl     # RBF核SVM模型(不同参数)
│   └── svm_rbf_C10_gammascale.pkl    # RBF核SVM模型(不同参数)
├── results/                       # 结果目录
├── my_model.keras                 # 保存的MLP模型
├── metrics.json                   # 模型性能指标
├── Fashion_MNIST_MLP_SVM_Comparison.ipynb  # 主要的Jupyter Notebook
└── Fashion_MNIST_MLP_SVM_Comparison.html   # HTML格式的笔记本
```

## 特性与功能

- **数据预处理**: 对图像数据进行归一化和转换，以适应模型训练
- **模型实现**:
  - **多层感知机 (MLP)**: 使用TensorFlow和Keras构建和训练深度神经网络
  - **支持向量机 (SVM)**: 使用Scikit-learn训练多种核函数的SVM模型
- **模型评估**: 使用准确率、精确度、召回率和F1分数进行全面评估
- **可视化**: 
  - 展示训练过程中的精度和损失曲线
  - 可视化分类结果和混淆矩阵
  - 模型性能对比图表

## 实验结果

根据metrics.json文件，MLP模型的性能指标如下：
- 训练集准确率: 88.6%
- 验证集准确率: 87.0%
- 训练集损失: 0.375
- 验证集损失: 0.427

项目中还比较了多种SVM模型的性能，包括使用不同的核函数（线性、多项式和RBF）和不同的超参数。

## 安装与使用

### 环境要求
- Python 3.6+
- TensorFlow 2.x
- Scikit-learn
- Numpy
- Pandas
- Matplotlib
- Seaborn
- Joblib

### 如何运行

1. 克隆此仓库:
```
git clone <repository-url>
```

2. 安装依赖:
```
pip install -r requirements.txt
```

3. 打开并运行Jupyter Notebook:
```
jupyter notebook Fashion_MNIST_MLP_SVM_Comparison.ipynb
```

## 主要发现

- MLP和SVM模型在Fashion MNIST数据集上都表现良好
- 通过调整超参数和模型结构，可以进一步提高性能
- 该项目提供了全面的代码框架，可用于图像分类任务的基准测试和比较

## 未来工作

- 实现更多的分类模型，如卷积神经网络(CNN)
- 尝试更多的特征工程技术
- 探索模型集成方法以提高分类性能

## 作者

程文展

## 许可证

[MIT](LICENSE)

## 参考资料

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Scikit-learn Documentation](https://scikit-learn.org/) 