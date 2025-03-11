# 卷积神经网络（LeNet）

 

## 定义网络

```python
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = torch.nn.Sequential(Reshape(), nn.Conv2d(1, 6, kernel_size=5,
                                               padding=2), nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2),
                          nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                          nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                          nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                          nn.Linear(120, 84), nn.Sigmoid(), nn.Linear(84, 10))
```



------

## 1. Reshape 层

```python
class Reshape(torch.nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)
```

- **作用：** 将输入的数据重塑（reshape），确保输入的形状是 `(batch_size, 1, 28, 28)`。
- **输入形状：** `(batch_size, 784)`，因为原始数据是扁平化的一维数据。
- **输出形状：** `(batch_size, 1, 28, 28)`，即 (通道数, 高度, 宽度)。

------

## 2. 第一个卷积层 Conv2d

```python
nn.Conv2d(1, 6, kernel_size=5, padding=2)
```

- **输入通道数：** 1（灰度图像）
- **输出通道数：** 6（设置的通道数）
- **卷积核大小：** 5×55 \times 5
- **填充：** padding=2
- **步幅：** 默认 stride=1

------

## 3. 激活函数 Sigmoid

```python
nn.Sigmoid()
```

- **作用：** 对卷积的结果进行非线性映射，但**不改变形状**。
- **输入形状：** `(batch_size, 6, 28, 28)`
- **输出形状：** `(batch_size, 6, 28, 28)`

------

## 4. 平均池化层 AvgPool2d

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

- **池化核大小：** 2×22 \times 2
- **步幅：** 2

------

## 5. 第二个卷积层 Conv2d

```python
nn.Conv2d(6, 16, kernel_size=5)
```

- **输入通道数：** 6
- **输出通道数：** 16
- **卷积核大小：** 5×55 \times 5
- **填充：** 无 padding，默认 stride=1

------

## 7. 第二个平均池化层 AvgPool2d

```python
nn.AvgPool2d(kernel_size=2, stride=2)
```

- **池化核大小：** 2×22 \times 2
- **步幅：** 2

------

## 8. 展平层 Flatten

```python
nn.Flatten()
```

- **作用：** 将三维数据（通道数、高度、宽度）展平成一维向量，用于输入到全连接层。
- **输入形状：** `(batch_size, 16, 5, 5)`

------

## ✅ 总结

| 层名                | 输入形状                   | 输出形状                   |
| ------------------- | -------------------------- | -------------------------- |
| **Reshape**         | `(batch_size, 784)`        | `(batch_size, 1, 28, 28)`  |
| **Conv2d(1→6)**     | `(batch_size, 1, 28, 28)`  | `(batch_size, 6, 28, 28)`  |
| **AvgPool2d**       | `(batch_size, 6, 28, 28)`  | `(batch_size, 6, 14, 14)`  |
| **Conv2d(6→16)**    | `(batch_size, 6, 14, 14)`  | `(batch_size, 16, 10, 10)` |
| **AvgPool2d**       | `(batch_size, 16, 10, 10)` | `(batch_size, 16, 5, 5)`   |
| **Flatten**         | `(batch_size, 16, 5, 5)`   | `(batch_size, 400)`        |
| **Linear(400→120)** | `(batch_size, 400)`        | `(batch_size, 120)`        |
| **Linear(120→84)**  | `(batch_size, 120)`        | `(batch_size, 84)`         |
| **Linear(84→10)**   | `(batch_size, 84)`         | `(batch_size, 10)`         |

------



## 训练过程

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)。"""
    def init_weights(m):#初始化参数
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)#应用参数
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()#将模型设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
```

------

## **函数签名及参数解析**

```python
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
```

### 参数：

- **`net`**：神经网络模型（`torch.nn.Module`）。
- **`train_iter`**：训练数据的 `DataLoader` 迭代器。
- **`test_iter`**：测试数据的 `DataLoader` 迭代器。
- **`num_epochs`**：训练的轮次（epoch），即数据集将被完整训练多少次。
- **`lr`**：学习率（learning rate）。
- **`device`**：设备（`cuda` 或 `cpu`），表示是否在 GPU 上运行。

------

##  **1. 初始化网络参数**

```python
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
```

### 解析：

- **`init_weights`** 是一个初始化函数，用于给网络中的 `Linear` 或 `Conv2d` 层赋予合适的初始权重。
- **`nn.init.xavier_uniform_`**：使用 Xavier 均匀分布初始化权重，适用于深度网络的权重初始化。
- **`net.apply(init_weights)`**：将该初始化函数应用到模型 `net` 的所有层上。

------

## **2. 将模型放置到指定设备**

```python
print('training on', device)
net.to(device)
```

### 解析：

- **`print`**：输出当前使用的设备（如 `cuda` 或 `cpu`）。
- **`net.to(device)`**：将模型移动到指定设备上（GPU 或 CPU）。

------

## **3. 定义优化器和损失函数**

```python
optimizer = torch.optim.SGD(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()
```

### 解析：

- **`optimizer`**：使用 **随机梯度下降 (SGD)** 优化器，学习率为 `lr`。
- **`net.parameters()`**：获取模型的可训练参数。
- **`loss`**：交叉熵损失函数，用于分类问题。

------

## **4. 定义动画器（可视化训练过程）**

```python
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
```

### 解析：

- `d2l.Animator`

   是 

  动手学深度学习 (d2l)

   提供的可视化工具，用于实时绘制：

  - 训练损失 (train loss)
  - 训练准确率 (train acc)
  - 测试准确率 (test acc)

- **`xlim=[1, num_epochs]`**：x 轴表示 epoch 的范围。

------

## **5. 训练循环**

### 5.1 初始化计时器和累加器

```python
timer, num_batches = d2l.Timer(), len(train_iter)
```

- **`timer`**：用于计时，计算每秒处理多少样本。
- **`num_batches`**：获取每个 epoch 的 mini-batch 数量。

------

### 5.2 进入训练循环

```python
for epoch in range(num_epochs):
    metric = d2l.Accumulator(3)
    net.train()
```

- **`for epoch in range(num_epochs)`**：循环 `num_epochs` 次。
- **`metric`**：用于累计 **训练损失**、**训练准确率**和**样本数量**。
- **`net.train()`**：将模型设置为训练模式。

------

### 5.3 小批量训练

```python
for i, (X, y) in enumerate(train_iter):
    timer.start()
    optimizer.zero_grad()
    X, y = X.to(device), y.to(device)
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    optimizer.step()
```

**循环内部工作流程：**

1. **`timer.start()`**：开始计时。
2. **`optimizer.zero_grad()`**：梯度清零，防止梯度累积。
3. **`X.to(device)`** & **`y.to(device)`**：将数据传入指定设备（如 GPU）。
4. **`y_hat = net(X)`**：前向传播，得到预测结果。
5. **`l = loss(y_hat, y)`**：计算损失。
6. **`l.backward()`**：反向传播，计算梯度。
7. **`optimizer.step()`**：更新模型参数。

------

### 5.4 计算损失和准确率

```python
with torch.no_grad():
    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
```

- **`torch.no_grad()`**：禁用梯度计算，减少内存使用。

- `metric.add()`

  ：

  - `l * X.shape[0]`：累计损失总和。
  - `d2l.accuracy(y_hat, y)`：计算准确率。
  - `X.shape[0]`：当前 batch 的样本数量。

------

### 5.5 更新可视化图表

```python
if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
    animator.add(epoch + (i + 1) / num_batches,
                 (train_l, train_acc, None))
```

- 每训练 1/5 的数据或最后一个 batch，更新可视化图表。

------

## **6. 每个 epoch 结束后验证测试集准确率**

```python
test_acc = evaluate_accuracy_gpu(net, test_iter)
animator.add(epoch + 1, (None, None, test_acc))
```

- **`evaluate_accuracy_gpu()`**：评估模型在测试集上的准确率。
- 将测试准确率添加到可视化图表中。

------

## **7. 输出训练结果**

```python
print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
      f'test acc {test_acc:.3f}')
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
```

- 打印最后一个 epoch 的损失、训练准确率、测试准确率。
- 计算每秒处理的样本数。

------

## **8. 计算处理速率**

```python
print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
      f'on {str(device)}')
```

- 如果使用 GPU，速度会非常快。

------

## ✅ **总结**

| 步骤               | 作用                                            |
| ------------------ | ----------------------------------------------- |
| **初始化网络**     | 初始化权重                                      |
| **将模型放入设备** | 使用 `to(device)` 将数据和模型迁移到 GPU 或 CPU |
| **优化器**         | 使用 SGD 进行优化                               |
| **前向传播**       | 计算预测值 `y_hat` 和损失 `l`                   |
| **反向传播**       | 计算梯度，并更新参数                            |
| **评估测试集**     | 使用 `evaluate_accuracy_gpu` 计算测试集准确率   |
| **可视化**         | 使用 `animator` 动态显示训练过程                |
| **计算处理速度**   | 计算每秒处理样本数                              |

