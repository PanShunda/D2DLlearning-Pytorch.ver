'''
沐神说过，微调模型/数据集的大小最好要十倍甚至百倍的大于你的目标模型/数据集，才能有比较好的微调效果。
'''

import os
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 注册hotdog数据集的下载信息到d2l数据源
d2l.DATA_HUB['hotdog'] = (d2l.DATA_URL + 'hotdog.zip',
                          'fba480ffa8aa7e0febbb511d181409f899b9baa5')
print(d2l.DATA_URL)  # 输出数据源基础URL

data_dir = d2l.download_extract('hotdog')  # 下载并解压hotdog数据集到指定路径

# 加载训练集和测试集图像数据
train_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'))
test_imgs = torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'))

# 展示前8个热狗和后8个非热狗的示例图像
hotdogs = [train_imgs[i][0] for i in range(8)]
not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=1.4)

# 定义图像标准化参数（使用ImageNet的均值和标准差）
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])

# 定义训练时的数据增强和标准化流程
# 图像增广
train_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),  # 随机裁剪为224x224
    torchvision.transforms.RandomHorizontalFlip(),  # 随机水平翻转
    torchvision.transforms.ToTensor(), normalize])

# 定义测试时的图像预处理流程（中心裁剪）
# 图像增广
test_augs = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),  # 缩放至256x256
    torchvision.transforms.CenterCrop(224),  # 中心裁剪为224x224
    torchvision.transforms.ToTensor(), normalize])

# 加载预训练的ResNet-18模型
# pretrained参数代表着是否使用预训练模型
pretrained_net = torchvision.models.resnet18(pretrained=True)

# 创建微调网络，替换最后一层全连接层为2类输出
finetune_net = torchvision.models.resnet18(pretrained=True)
# 重新创建并替换了最后一层全连接层，将输出层改为2类输出。
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
nn.init.xavier_uniform_(finetune_net.fc.weight)  # 初始化新添加的全连接层权重

def train_fine_tuning(net, learning_rate, batch_size=128, num_epochs=5, param_group=True):
    """训练模型函数，支持微调和从头开始训练。

    Args:
        net (nn.Module): 要训练的神经网络模型。
        learning_rate (float): 学习率，预训练参数的小，新层的大（若param_group为True）。
        batch_size (int): 数据加载的批量大小，默认128。
        num_epochs (int): 训练的轮数，默认5。
        param_group (bool): 是否为不同参数组设置不同学习率，默认True。

    Returns:
        None
    """
    # 创建训练集数据迭代器（应用数据增强）
    train_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_augs),
        batch_size=batch_size, shuffle=True)

    # 创建测试集数据迭代器（应用标准化处理）
    test_iter = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=test_augs),
        batch_size=batch_size)

    # 使用所有可用GPU设备进行训练
    devices = d2l.try_all_gpus()
    loss = nn.CrossEntropyLoss(reduction="none")  # 定义交叉熵损失函数

    # 根据是否分组设置优化器参数
    if param_group:
        # 预训练层使用较小学习率，新层fc使用较大学习率（初始学习率的10倍）
        # 主要是因为预训练层已经训练好，不需要再训练，而新层（最后一层）需要训练，所以需要较大的学习率
        params_1x = [param for name, param in net.named_parameters() if name not in ["fc.weight", "fc.bias"]]
        # PyTorch 支持在同一个优化器中使用不同的学习率
        trainer = torch.optim.SGD([
            {'params': params_1x},
            {'params': net.fc.parameters(), 'lr': learning_rate * 10}
        ], lr=learning_rate, weight_decay=0.001)
    else:
        # 所有参数使用统一学习率
        trainer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)

    # 执行训练流程（使用d2l的训练函数）
    d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)

'''
使用较小的学习率进行微调（预训练参数学习率5e-5，新层5e-4）
'''
train_fine_tuning(finetune_net, 5e-5)

'''
为了进行比较，所有模型参数初始化为随机值（从头开始训练）
'''
scratch_net = torchvision.models.resnet18()
scratch_net.fc = nn.Linear(scratch_net.fc.in_features, 2)
train_fine_tuning(scratch_net, 5e-4, param_group=False)
