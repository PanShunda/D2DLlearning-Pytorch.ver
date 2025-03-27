import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

class Residual(nn.Module):
    """残差块模块。

    参数:
    input_channels (int): 输入张量的通道数。
    num_channels (int): 残差块的输出通道数。
    use_1x1conv (bool, 可选): 是否使用1x1卷积调整输入维度以匹配输出维度。默认False。
    strides (int, 可选): 卷积层的步长。默认1。

    forward方法:
    输入:
    X (torch.Tensor): 输入张量，形状为(batch_size, input_channels, height, width)。

    返回:
    torch.Tensor: 经过残差块处理后的输出张量，形状为(batch_size, num_channels, height_out, width_out)。
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False,
                 strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3,
                               padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3,
                               padding=1)
        # 残差块中的第三个卷积层，可选是否使用1x1卷积调整输入维度以匹配输出维度。
        # 这就是那个残差1x1conv。
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X):
        """前向传播方法，执行残差块的计算流程。

        参数:
        X (torch.Tensor): 输入张量。

        返回:
        torch.Tensor: 经过残差块处理后的输出张量。
        """
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

blk = Residual(3, 3)
X = torch.rand(4, 3, 6, 6)
Y = blk(X)
print(Y.shape)

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)
# 初始卷积层：7x7卷积，步长2，随后是BN、ReLU和最大池化层，用于特征提取和下采样。

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    """生成包含多个残差块的序列模块。

    参数:
    input_channels (int): 输入通道数。
    num_channels (int): 每个残差块的输出通道数。
    num_residuals (int): 当前阶段包含的残差块数量。
    first_block (bool, 可选): 是否为第一个残差块阶段（用于处理初始层后的输入调整）。默认False。

    返回:
    nn.Sequential: 包含多个Residual模块的序列。
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(
                Residual(input_channels, num_channels, use_1x1conv=True,
                         strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=False))
# 第一个残差块组：输入64通道，输出64通道，包含2个残差块，作为第一个阶段。
b3 = nn.Sequential(*resnet_block(64, 128, 2))
# 第二个残差块组：输入64通道，输出128通道，包含2个残差块，进行下采样。
b4 = nn.Sequential(*resnet_block(128, 256, 2))
# 第三个残差块组：输入128通道，输出256通道，包含2个残差块，进行下采样。
b5 = nn.Sequential(*resnet_block(256, 512, 2))
# 第四个残差块组：输入256通道，输出512通道，包含2个残差块，进行下采样。

'''
输入 → [b1] → [b2] → [b3] → [b4] → [b5] → 池化 → 展平 → 全连接 → 输出
尺寸变化：
224x224 → 56x56 → 56x56 → 28x28 → 14x14 → 7x7 → 1x1 → 10
'''


net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化，将特征图压缩为1x1
    nn.Flatten(),  # 展平为一维向量
    nn.Linear(512, 10)  # 全连接层输出10类别
)
# 完整的ResNet模型结构，包含初始卷积层、四个残差块组、全局池化和分类层。

X = torch.rand(size=(1, 1, 224, 224))
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
# 验证网络结构，通过随机输入推断各层输出形状。

lr, num_epochs, batch_size = 0.05, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=96)
# 加载Fashion-MNIST数据集，调整图像尺寸为96x96。

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
# 使用GPU训练模型，执行10个epoch，学习率0.05。
