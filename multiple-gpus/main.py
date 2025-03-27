import torch
from torch import nn
from d2l import torch as d2l


def resnet18(num_classes, in_channels=1):
    """构建一个稍加修改的ResNet-18模型。

    参数:
    num_classes (int): 输出类别数。
    in_channels (int): 输入图像的通道数，默认为1（灰度图像）。

    返回:
    torch.nn.Module: 构建的ResNet-18模型实例。
    """
    def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
        """构建残差块序列。

        参数:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数。
        num_residuals (int): 残差块的数量。
        first_block (bool): 是否是第一个残差块，用于调整初始层的步长和卷积。

        返回:
        torch.nn.Sequential: 包含多个残差块的序列。
        """
        blk = []
        # 循环添加残差块，第一个非初始块使用步长2的卷积层以减半特征图尺寸
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(
                    d2l.Residual(in_channels, out_channels, use_1x1conv=True,
                                 strides=2))
            else:
                blk.append(d2l.Residual(out_channels, out_channels))
        return nn.Sequential(*blk)

    # 初始化卷积层、批量归一化和激活函数
    net = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU())

    # 添加四个残差块模块
    net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
    # 第二个残差块通道数翻倍，特征图尺寸减半
    net.add_module("resnet_block2", resnet_block(64, 128, 2))
    # 第三个残差块继续通道翻倍并减半尺寸
    net.add_module("resnet_block3", resnet_block(128, 256, 2))
    # 最后一个残差块通道翻倍并减半尺寸
    net.add_module("resnet_block4", resnet_block(256, 512, 2))

    # 全局平均池化将特征图压缩为1x1尺寸
    net.add_module("global_avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
    # 全连接层：展平输出并分类到num_classes类别
    net.add_module("fc",
                   nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes)))
    return net

net = resnet18(10)
devices = d2l.try_all_gpus()


def train(net, num_gpus, batch_size, lr):
    """训练模型使用多GPU并行。

    参数:
    net (torch.nn.Module): 待训练的神经网络模型。
    num_gpus (int): 使用的GPU数量。
    batch_size (int): 训练和测试的批量大小。
    lr (float): 学习率。

    返回:
    None: 直接在训练过程中输出结果和绘制准确率曲线。
    """
    # 加载Fashion-MNIST数据集，预处理并分批次
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 获取可用的GPU设备列表
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        """初始化权重函数，使用正态分布初始化全连接和卷积层的权重"""
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.normal_(m.weight, std=0.01)

    # 应用权重初始化
    net.apply(init_weights)

    # 将模型包装为DataParallel以支持多GPU并行
    net = nn.DataParallel(net, device_ids=devices)
    # 定义优化器和损失函数
    trainer = torch.optim.SGD(net.parameters(), lr)
    loss = nn.CrossEntropyLoss()

    # 初始化计时器和动画绘图器
    timer, num_epochs = d2l.Timer(), 10
    animator = d2l.Animator('epoch', 'test acc', xlim=[1, num_epochs])

    for epoch in range(num_epochs):
        net.train()
        timer.start()
        # 训练迭代：前向传播、计算损失、反向传播、参数更新
        for X, y in train_iter:
            trainer.zero_grad()
            X, y = X.to(devices[0]), y.to(devices[0])
            l = loss(net(X), y)
            l.backward()
            trainer.step()

        timer.stop()
        # 评估测试集准确率并更新动画
        animator.add(epoch + 1, (d2l.evaluate_accuracy_gpu(net, test_iter),))

    # 输出最终结果和时间统计
    print(f'test acc: {animator.Y[0][-1]:.2f}, {timer.avg():.1f} sec/epoch '
          f'on {str(devices)}')


train(net, num_gpus=1, batch_size=256, lr=0.1)
train(net, num_gpus=2, batch_size=512, lr=0.2)
