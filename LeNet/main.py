

import torch
from torch import nn
from d2l import torch as d2l
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28)

net = nn.Sequential(
    Reshape(),
    nn.Conv2d(1, 6, 11, padding=5),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(6, 16, 5),
    nn.Sigmoid(),
    nn.AvgPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, nn.Sequential):
        net.eval()
        if device is None:
            device = next(iter(net.parameters())).device
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """训练神经网络模型（第6章实现）

    参数：
    net (torch.nn.Module): 神经网络模型
    train_iter (DataLoader): 训练数据迭代器
    test_iter (DataLoader): 测试数据迭代器
    num_epochs (int): 训练轮数
    lr (float): 学习率
    device (torch.device): 训练设备（如'cpu'或'cuda'）

    返回：
    无（通过打印输出结果）
    """
    def init_weights(m):
        """使用Kaiming正态初始化初始化线性层权重"""
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
    net.apply(init_weights)  # 应用权重初始化

    print('Training on', device)
    net.to(device)  # 将网络移动到指定设备
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # 创建SGD优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 初始化训练过程可视化器
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)  # 计时器和批次数量

    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)  # 累积损失、准确率、样本数
        net.train()  # 设置为训练模式
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()  # 梯度清零
            X, y = X.to(device), y.to(device)  # 数据移动到设备
            y_pred = net(X)  # 前向传播
            l = criterion(y_pred, y)  # 计算损失
            l.backward()  # 反向传播
            optimizer.step()  # 参数更新

            # 更新训练指标（在梯度上下文外执行）
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_pred, y), X.shape[0])
            timer.stop()

            # 计算当前epoch的平均损失和准确率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]

            # 每隔num_batches/5个批次或最后一个批次时更新可视化
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                plt.pause(0.001)  # 手动刷新绘图窗口

        # 计算测试集准确率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))  # 添加测试结果到动画

    # 输出最终结果和速度统计
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')


lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
plt.show()  # 保持窗口打开