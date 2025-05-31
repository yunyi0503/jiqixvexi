# Fashion MNIST 图像分类

本项目使用 TensorFlow 和 Keras 实现了针对 Fashion MNIST 数据集的图像分类模型。通过卷积神经网络(CNN)的应用，实现了对服装图像的高精度分类。

## 项目概述

Fashion MNIST 数据集是由 Zalando Research 发布的替代传统 MNIST 手写数字数据集的现代数据集。它包含 70,000 张服装物品的灰度图像，分为 10 个类别。本项目展示了如何构建、训练和评估卷积神经网络 (CNN) 来准确分类这些服装物品，并提供了完整的数据处理、模型构建、训练和评估的工作流程。

## 数据集详情

Fashion MNIST 包括：
- 60,000 张训练图像和 10,000 张测试图像
- 10 个服装类别，每个类别有 7,000 张图像
- 类别标签：
  - 0: T恤/上衣 (T-shirt/top)
  - 1: 裤子 (Trouser)
  - 2: 套头衫 (Pullover)
  - 3: 连衣裙 (Dress)
  - 4: 外套 (Coat)
  - 5: 凉鞋 (Sandal)
  - 6: 衬衫 (Shirt)
  - 7: 运动鞋 (Sneaker)
  - 8: 包 (Bag)
  - 9: 短靴 (Ankle boot)
- 每张图像为 28x28 像素的灰度图，像素值范围为 0-255
- 相比传统MNIST数据集，Fashion MNIST提供了更具挑战性的分类任务

## 项目内容

- `fashion_mnist.ipynb`：包含完整实现和详细注释的 Jupyter 笔记本，包括数据加载、探索性数据分析、模型构建、训练和评估的全过程
- `fashion_mnist_cnn_model.keras`：在 Fashion MNIST 数据集上训练的已保存CNN模型，可直接加载使用
- `fashion_mnist(8).html`：笔记本的 HTML 导出版本，方便不需要运行代码的用户查看项目内容和结果

## 实现方法

### 数据预处理
- 数据加载：使用 `keras.datasets` 加载 Fashion MNIST 数据集
- 数据归一化：将像素值从 0-255 缩放到 0-1 范围，提高模型训练效率
- 数据重塑：对于CNN模型，将原始形状 (28, 28) 重塑为 (28, 28, 1)，添加通道维度
- 标签独热编码：将类别标签转换为独热编码形式

### 模型架构
CNN模型架构包括：
- 卷积层：使用多个卷积层提取图像特征，每层使用不同数量的过滤器
- 池化层：使用最大池化减少特征图尺寸并保留重要特征
- Dropout层：添加防止过拟合的随机失活层
- 全连接层：将卷积特征映射到分类空间
- 输出层：10个神经元对应10个类别，使用softmax激活函数

详细结构：
```
模型: Sequential
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
                                                                 
 max_pooling2d (MaxPooling2D)  (None, 13, 13, 32)      0         
                                                                 
 conv2d_1 (Conv2D)           (None, 11, 11, 64)        18496     
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 5, 5, 64)         0         
 2D)                                                             
                                                                 
 conv2d_2 (Conv2D)           (None, 3, 3, 128)         73856     
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dropout (Dropout)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 128)               147584    
                                                                 
 dense_1 (Dense)             (None, 10)                1290      
                                                                 
=================================================================
总参数：241,546
可训练参数：241,546
不可训练参数：0
```

### 训练过程
- 优化器：使用Adam优化器，学习率为0.001
- 损失函数：分类交叉熵
- 批量大小：64
- 训练轮次：10-20轮
- 验证：使用20%的训练数据作为验证集
- 回调函数：使用EarlyStopping和ModelCheckpoint监控训练过程

### 评估指标
- 准确率：测试集上的分类准确率
- 混淆矩阵：展示各类别之间的误分类情况
- 分类报告：包含每个类别的精确率、召回率和F1分数
- 可视化：错误分类样本的可视化分析

## 环境需求

运行此项目需要：
- Python 3.6+
- TensorFlow 2.3+
- Keras (TensorFlow集成版)
- NumPy 1.18+
- Pandas 1.0+
- Matplotlib 3.2+
- Seaborn 0.10+
- Scikit-learn 0.22+
- Jupyter Notebook/Lab

可以通过以下命令安装所需依赖：
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn jupyter
```

## 使用方法

### 运行完整项目
1. 克隆此仓库：
   ```bash
   git clone  https://github.com/yunyi0503/jiqixvexi.git
   ```

2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt 
   ```

3. 打开并运行 Jupyter 笔记本：
   ```bash
   jupyter notebook fashion_mnist.ipynb
   ```

### 使用预训练模型进行预测
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 加载模型
model = tf.keras.models.load_model('fashion_mnist_cnn_model.keras')

# 加载测试数据
(_, _), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 数据预处理
x_test = x_test.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# 类别名称
class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套',
               '凉鞋', '衬衫', '运动鞋', '包', '短靴']

# 预测单个图像
def predict_image(image_index):
    # 获取图像
    img = x_test[image_index]
    # 扩展维度以适应模型输入
    img_array = tf.expand_dims(img, 0)
    # 进行预测
    predictions = model.predict(img_array)
    # 获取预测类别
    predicted_class = np.argmax(predictions[0])
    true_class = y_test[image_index]
    
    # 显示图像和预测结果
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(img.reshape(28,28), cmap='gray')
    plt.title(f'真实类别: {class_names[true_class]}')
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.bar(range(10), predictions[0])
    plt.xticks(range(10), class_names, rotation=90)
    plt.title(f'预测类别: {class_names[predicted_class]}')
    plt.tight_layout()
    plt.show()
    
# 示例使用
predict_image(42)  # 预测第42个测试图像
```

## 结果

CNN 模型在 Fashion MNIST 测试集上取得了约93%的准确率，展示了对服装物品特征和模式的有效学习。不同类别的性能各异：
- 裤子、运动鞋和短靴等形状独特的物品分类效果最佳(>95%)
- T恤/上衣与衬衫之间存在一定的混淆，这也是最具挑战性的分类任务

通过混淆矩阵分析，我们可以看到主要的混淆发生在视觉上相似的类别之间，如衬衫与T恤，外套与套头衫等。

## 未来改进

可能的改进方向包括：
- 使用更复杂的模型架构，如ResNet或DenseNet
- 应用数据增强技术增加训练样本的多样性
- 尝试不同的优化器和学习率策略
- 使用迁移学习，在更大的数据集上预训练模型

## 许可

本项目仅供教育和研究目的使用。Fashion MNIST 数据集由 Zalando Research 提供，根据MIT许可证分发。 