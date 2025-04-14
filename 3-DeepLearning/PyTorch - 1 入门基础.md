


## 张量 Tensor 


张量是一个<font style="background-color:#FBDE28;color:black">更高维度的数组</font>。

| 名称 | 维度      | 举例                          |
|------|-----------|-------------------------------|
| 标量 | 0 维      | 3.14                          |
| 向量 | 1 维      | [1, 2, 3]                     |
| 矩阵 | 2 维      | `[[1, 2], [3, 4]]`               |
| 张量 | 3 维及以上 | `[[[1,2],[3,4]], [[5,6],[7,8]]] `|

**向量 和 矩阵 **都是张量的特例。



Numpy 库更优化，但 PyTorch 的张量有“超能力”，比如可以在 GPU 上执行更快的操作。

![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743655261892-a1720194-513a-47a0-835b-cbdf1b7b0f43.png)![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743655124979-dc5db0b5-ac9a-4066-95d8-bbff96a3b7c0.png)

#### 访问元素 
![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743655186733-d6027f09-b227-4af8-8baa-4c612b2bf5e9.png)![](https://cdn.nlark.com/yuque/0/2025/png/295096/1743655204278-3f5ab79e-5d77-4cbf-bd1c-1d4b49aaf1c0.png)

#### torch 张量存储方法
![](https://cdn.nlark.com/yuque/0/2025/png/295096/1744031518584-4a4c6a72-91c3-4ac2-a033-f3e79927209d.png)![](https://cdn.nlark.com/yuque/0/2025/png/295096/1744094066269-a310bc7e-c721-467a-8728-642efbd46d5a.png)
张量中的值被分配到由 torch.Storage 实例所管理的<font style="background-color:#FBDE28;color:black">连续内存块</font>中。

#### 张量的元数据

+ **维度系统** `x.ndimension()`
+ **是否连续** `x.is_condiguous()`

当张量被转置，reshape, 切片，permute 交换维度后，可能不连续

+ **存在哪儿** x.device
+ **大小**
描述张量在每个维度的长度，如下四阶张量，Size描述张量的  batch=6，channels=3，rows=4，cols=4

`x = torch.randn(6,3,4,4)`

+ **步长 stride**

在内存中访问张量元素时，每个维度间隔的步长，即从一个元素到下一个元素在内存中所需跳过的元素个数。

`x = torch.randn(2,3,4).stride()` ==> (12,4,1)

从通道1访问到通道2时，需要访问3*4个元素； 从行1访问到行2时，需要访问4个元素

最后一阶的步长始终是 1

+ **偏移量 offset**

张量数据在存储缓冲区中的起始位置


#### 张量的运算

在 PyTorch 中，张量的加减乘除（也称为元素级操作）需要满足一定的条件才能进行运算。并不是任意两个张量都可以直接进行运算，主要有以下几个条件需要满足：

1. **形状一致性（Shape Compatibility）**  
 **广播（Broadcasting）**：在进行元素级运算时，如果两个张量的形状不同，PyTorch 会尝试使用广播机制（broadcasting）来自动扩展张量的形状以使其匹配。广播规则决定了两个张量如何通过扩展形状来使它们的维度对齐。  
 •	如果形状不兼容且无法广播，运算将失败。

**广播规则**：广播机制会对形状不同的张量进行自动扩展，具体规则如下：  
    •	从右向左对齐维度。  
    •	如果一个张量在某个维度的大小为 1，而另一个张量在该维度上有更大的尺寸，则大小为 1 的维度会被扩展为匹配更大维度的大小。  
    •	如果某个维度在两个张量中都有非 1 的大小，但它们的大小不同，则无法广播，运算会失败。

2. **数据类型一致性（Type Compatibility）**  
 •	进行元素级运算的两个张量必须具有兼容的数据类型。如果数据类型不兼容，PyTorch 会尝试进行类型转换，但某些类型可能无法直接转换。  
 •	例如，一个 float32 张量和一个 int64 张量进行加法时，PyTorch 会将 int64 转换为 float32，然后进行加法。
3. **运算规则**  
 •	加法（+）：两个张量可以进行加法运算，如果它们的形状相同或可以通过广播机制对齐。  
 •	减法（-）：同样，两个张量可以进行减法运算，条件是形状一致或可广播。  
 •	乘法（*）：乘法是逐元素乘法，如果形状一致或能够广播。  
 •	除法（/）：除法是逐元素除法，要求形状一致或可以广播。

```python
z = torch.matmul(x, y) #矩阵乘法
z = torch.mm(x, y) # 2D矩阵乘法
z = torch.dot(x, y) # 计算张量的点积
z = torch.prod(x) # 所有元素相乘
```

#### 张量的 API
+ 从 np 转换  `x=torch.from_numpy(np.array([1,2,3])).float()`
+ 转置`torch.transpose(x, 0, 1)`
+ 范数 `torch.norm(x)`
    - **L1 范数（Manhattan Norm）** 各个元素绝对值之和
    - **L2 范数（Euclidean Norm）** 向量各个元素的平方和的平方根 $\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + … + x_n^2}$
    - **Frobenius 范数** 矩阵中所有元素的平方和的平方根 $\|A\|F = \sqrt{\sum{i,j} |a_{ij}|^2}$
+ 标准差 `torch.std(x)`

### 使用张量表征真实数据
#### 处理图像
计算机看不懂图片，只能识别数字。因此，我们需要把图片转成张量。

1️⃣ 黑白图像
一个<font style="background-color:#FBDE28;color:black">二维矩阵</font>，每个像素是0（黑）~255（白）的数值
$$\begin{vmatrix}
95 & 130 & 0 & 255 \\
208 & 152 & 172 & 159 \\
42 & 133 &  118 &  95\\
197&  58& 142&  86
\end{vmatrix}$$
4*4 像素的黑白图片，本质是一个 4*4 矩阵
![](https://cdn.nlark.com/yuque/0/2025/png/295096/1744184691176-325d748e-8310-472b-a8f3-1e3bfb126db2.png)

2️⃣ **彩色图像 RGB **

本质是 三阶张量

![](https://cdn.nlark.com/yuque/0/2025/png/295096/1744184912859-b59c0bf1-cc26-44b7-8818-4da99cfd4f2a.png)
```python
[
  [[255, 0, 0], [0, 255, 0]],   # 红 | 绿
  [[0, 0, 255], [255, 255, 255]] # 蓝 | 白
]
```
其中每个三元组，代表一个颜色像素

❓为什么是3通道？  
人眼有感知红绿蓝的视锥细胞，显示器也基于RGB发光原理。

3️⃣ 加载图片到张量

```python
from PIL import Image
import numpy as np

# 1. 加载图片 → Python图像对象
img = Image.open("/Users/jimmy/Desktop/WechatIMG85.jpg")

# 2. 转换成NumPy数组（即张量）
img_array = np.array(img)  # 形状：(高度, 宽度, 3)

print(img_array.shape)  # 输出例如 (480, 640, 3)
print(img_array[0,0])   # 查看左上角第一个像素的RGB值，例如 [255, 240, 218]
```

3️⃣ **归一化数据**

为什么要做？

+ 原始像素值0-255范围较大，直接计算可能导致数值不稳定（像用米和毫米混合计算容易出错）
+ 许多激活函数（如Sigmoid）在0-1范围内工作更好

常用方法：

1. 除255法：`x = x / 255.0` → 得到0-1范围
2. 均值标准化：`x = (x - 127.5) / 127.5` → 得到-1到1范围


```python
# 原始像素值
print(img_array[0,0])  # 输出 [148, 120, 96]

# 方法1：除255
normalized = img_array / 255.0
print(normalized[0,0])  # 输出 [0.58, 0.47, 0.38]

# 方法2：均值标准化
normalized = (img_array - 127.5) / 127.5
print(normalized[0,0])  # 输出 [0.16, -0.06, -0.25]
```

#### 表征表格数据
**1️⃣ 加载**[**葡萄酒数据集**](https://archive.ics.uci.edu/dataset/109/wine)**为张量**

| 酒精浓度 | 酸度 | 酚类 | 评分 |
|----------|------|------|------|
| 13.2     | 0.8  | 2.1  | 7    |
| 12.5     | 0.6  | 1.9  | 6    |


```python
import pandas as pd
import torch

# 加载数据集（假设是CSV文件）
data = pd.read_csv('wine.csv')
print(data.head())  # 查看前5行

# 提取特征和标签
features = data[['酒精浓度', '酸度', '酚类']].values  # NumPy数组
labels = data['评分'].values

# 转换为张量
X = torch.tensor(features, dtype=torch.float32)  # 形状: (样本数, 特征数)
y = torch.tensor(labels, dtype=torch.long)       # 形状: (样本数,)
print("特征张量形状:", X.shape)
print("标签张量形状:", y.shape)
```

**2️⃣ 表示分数**

评分是典型的分类问题，需要处理为离散值：

+ 直接使用：如果评分是连续数值（如7.2分），可直接作为回归任务的标签。
+ 离散化：如果评分是整数等级（如1-10分），需按分类任务处理。

3️⃣ 独热编码（One-Hot Encoding）

当特征或标签是类别型数据（如葡萄品种：赤霞珠/梅洛/西拉），需转换为数值：

```python
from sklearn.preprocessing import OneHotEncoder

# 假设有一列'葡萄品种'包含类别
varieties = data[['葡萄品种']]  # 形状: (样本数, 1)

# 独热编码
encoder = OneHotEncoder()
encoded_varieties = encoder.fit_transform(varieties).toarray()  # 稀疏矩阵转密集数组
encoded_tensor = torch.tensor(encoded_varieties, dtype=torch.float32)
print("编码后形状:", encoded_tensor.shape)  # 例如 torch.Size([178, 3])（3个品种）
```

4️⃣ 何时分类？何时回归？

分类任务：标签是离散类别（如葡萄酒等级A/B/C）。

`criterion = torch.nn.CrossEntropyLoss()`

回归任务：标签是连续数值（如精确评分7.2分）。

`criterion = torch.nn.MSELoss()`

5️⃣ 寻找阈值

对于二分类问题（如“优质葡萄酒”是/否），需将连续分数转换为二元标签：

```python
# 假设评分≥7为优质葡萄酒
threshold = 7
binary_labels = (labels >= threshold).astype(int)  # NumPy数组
binary_tensor = torch.tensor(binary_labels)       # 转换为张量
```
#### 表征时间序列数据
时间序列数据需要显式保留时间顺序信息，通常表示为 `[时间步长, 特征数]` 的张量。

| 时间 | 温度 | 湿度 | 压力 |
|------|------|------|------|
| t1   | 23   | 40   | 1012 |
| t2   | 24   | 42   | 1010 |
| t3   | 22   | 38   | 1013 |
```python
[
  [23, 40, 1012],
  [24, 42, 1010],
  [22, 38, 1013]
] # shape: (3, 3)
```


#### 表征文本数据
🔣 1. 文本编码为数字（tokenization）

常见方式：

- 字符级编码：每个字符一个编号
- 词级编码：每个词一个编号
- 子词编码（如 BPE, WordPiece）：将词拆分为子词
- 使用现成 tokenizer：如 Hugging Face 的 tokenizer

```python
text = "hello world"
vocab = {"hello": 0, "world": 1}
token_ids = [vocab[word] for word in text.split()]
```

**📦 2. 转换为 PyTorch 张量**

```python
import torch

token_ids = [0, 1]
tensor = torch.tensor(token_ids)
print(tensor)  # tensor([0, 1])
```

🧠 3. Word Embedding 单词嵌入

word embedding 是将文本**映射为 连续的、稠密的向量**表示（float型张量） 的一种技术，是从 “token id 张量” 到 “语义向量张量” 的关键步骤。

```python
import torch
import torch.nn as nn

# 假设我们有 10 个词，embedding 维度是 5
embedding = nn.Embedding(num_embeddings=10, embedding_dim=5)

# 输入是词 id，比如句子是 [1, 2, 4]
input_ids = torch.tensor([1, 2, 4])  # shape: (3,)

# 得到词向量
embedded = embedding(input_ids)
print(embedded.shape)  # torch.Size([3, 5])
```



## 自动微分 (autograd)
回归 深度学习的核心思想：

模型训练的本质 = 不断更新参数（<font style="background-color:#FBDE28;color:black">权重和偏置</font>），使 <font style="background-color:#FBDE28;color:black">损失函数最小化</font>

—— 所以：我们要找<font style="background-color:#FBDE28;color:black">最优解（最小值）</font>

—— 找最小值：要用<font style="background-color:#FBDE28;color:black">梯度下降</font>

—— 梯度下降：要先求导

—— 求导过程：由 <font style="background-color:#FBDE28;color:black">自动微分/自动求导机制</font> 完成


自动微分是一种计算导数的技术，它通过记录**前向传播**过程中每个操作的<font style="background-color:#FBDE28;color:black">中间结果和操作本身</font>，然后反向逐步计算各个变量对目标函数的导数（<font style="background-color:#FBDE28;color:black">梯度</font>）。

<font style="background-color:#FBDE28;color:black">反向传播是实现自动求导的一种算法</font>（在神经网络中）。

#### 代码示例
```python
import torch

# 创建一个需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2  # y = x^2

# 计算梯度
y.backward()  # dy/dx = 2x
print(x.grad)  # 输出: tensor(4.) (因为 x=2, 2*2=4)
```

以下打开 github 代码查看：

+ 动态计算图
+ 梯度积累和清零
+ 阻止梯度跟踪
+ 高阶梯度

## 简单神经网络
