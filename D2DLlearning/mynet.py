import torch
import numpy as np
from torch.utils import data
from d2l import torch as d2l
from torch import nn

# 先生成一个数据集
## 训练的目标权值
true_w = torch.tensor([2,-3.4])
true_b = 4.2
# 这里生成了实际值
features,labels = d2l.synthetic_data(true_w, true_b,1000)

# 读取数据集
## 定义一个函数 需要一个数组元组 一个批次大小 一个布尔值来说明是否为训练集 返回一个dataloader
def load_array(data_arrays,batch_size,is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=is_train)
## 数据集
batch_size = 10
data_iter = load_array((features,labels),batch_size)

## 此时可以先获取data迭代器的第一项先看看
## print(next(iter(data_iter)))


# 生成一层线性网络，输入2，输出1（暂时未初始化）
net = nn.Sequential(nn.Linear(2,1))
# 初始化 normal_的意思是使用0，0.01的正态分布替换weight（权重）的值
# 初始化 第二行是将偏置全部设置为0
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(),lr=0.03)

#开始训练
num_epochs = 5
for epoch in range(num_epochs):
    for x,y in data_iter:
        l = loss(net(x),y)
        #范式 记清楚就行 必须得将梯度清空再计算梯度 再利用梯度进行优化
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features),labels)
    print('Epoch %d, Loss %.4f' % (epoch+1, l))

# 输出：
# Epoch 1, Loss 0.0002
# Epoch 2, Loss 0.0001
# Epoch 3, Loss 0.0001
# Epoch 4, Loss 0.0001
# Epoch 5, Loss 0.0001

# 计算一下误差
# w和b表示的是训练结束后真实的参数
# true_w和b表示的是我们最开始定义的数据集参数
w = net[0].weight.data
b = net[0].bias.data
print(true_w - w.reshape(true_w.shape))
print(true_b - b)

# tensor([ 1.0014e-05, -4.2534e-04])
# tensor([-0.0005])