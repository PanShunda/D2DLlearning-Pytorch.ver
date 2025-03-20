import torch
from torch import nn
from d2l import torch as d2l

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    """批量归一化函数。

    参数:
    X (torch.Tensor): 输入张量，形状为(批量大小, 特征数)或(批量大小, 通道数, 高, 宽)
    gamma (torch.Tensor): 缩放参数，形状与X的通道/特征维度一致
    beta (torch.Tensor): 偏移参数，形状与X的通道/特征维度一致
    moving_mean (torch.Tensor): 移动平均均值，用于推理阶段
    moving_var (torch.Tensor): 移动平均方差，用于推理阶段
    eps (float): 平滑项，防止除零错误
    momentum (float): 动量系数，控制移动平均的更新速度

    返回:
    Y (torch.Tensor): 归一化并缩放后的输出张量
    moving_mean.data (torch.Tensor): 更新后的移动平均均值（无梯度）
    moving_var.data (torch.Tensor): 更新后的移动平均方差（无梯度）
    """

    if not torch.is_grad_enabled():  # 推理阶段：使用移动统计量
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:  # 训练阶段：计算当前batch的统计量
        assert len(X.shape) in (2, 4), "仅支持全连接层(2D)和卷积层(4D)"

        if len(X.shape) == 2:  # 全连接层处理
            mean = X.mean(dim=0)
            var = ((X - mean)**2).mean(dim=0)
        else:  # 卷积层处理（跨空间维度计算）
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean)**2).mean(dim=(0, 2, 3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)  # 当前batch归一化

        # 更新移动统计量（指数加权平均）
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var

    Y = gamma * X_hat + beta  # 线性变换恢复表达能力
    return Y, moving_mean.data, moving_var.data

class BatchNorm(nn.Module):
    def __init__(self, num_features, num_dims):
        """初始化层参数和统计量。

        Args:
            num_features (int): 输入张量的特征数量（通道数）
            num_dims (int): 输入张量的维度（2表示全连接层，4表示卷积层）
        """
        super().__init__()

        # 根据输入维度设置参数形状
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        # 初始化缩放参数 gamma 和偏移参数 beta
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))

        # 初始化移动均值和方差，用于在推理时跟踪统计信息
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)


    def forward(self, X):
        """执行批归一化前向传播。

        参数：
            X (Tensor): 输入特征张量。

        返回：
            Tensor: 归一化后的输出张量。
        """
        # 确保移动平均值和方差与输入张量在相同设备上
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # 执行批归一化并更新移动平均值和方差
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean, self.moving_var,
            eps=1e-5, momentum=0.9)
        return Y

"""应用batchnorm于lenet中"""
net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4),
                    nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16,
                              kernel_size=5), BatchNorm(16, num_dims=4),
                    nn.Sigmoid(), nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Flatten(), nn.Linear(16 * 4 * 4, 120),
                    BatchNorm(120, num_dims=2), nn.Sigmoid(),
                    nn.Linear(120, 84), BatchNorm(84, num_dims=2),
                    nn.Sigmoid(), nn.Linear(84, 10))

