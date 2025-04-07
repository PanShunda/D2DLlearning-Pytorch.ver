import collections
import math
import os
import shutil
import pandas as pd
import torch
import torchvision
from torch import nn
from d2l import torch as d2l

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 使用交互式后端

demo = False

data_dir = './data/'


#@save
def read_csv_labels(fname):
    """读取CSV文件并返回文件名到标签的字典映射。

    参数:
    fname (str): CSV文件路径，文件格式为'filename,label'。

    返回:
    dict: 键为文件名（不带扩展名），值为对应标签的字典。
    """
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]  # 跳过标题行
    tokens = [l.rstrip().split(',') for l in lines]
    return {name: label for name, label in tokens}


#@save
def copyfile(filename, target_dir):
    """将文件复制到目标目录并自动创建目录。

    参数:
    filename (str): 源文件路径。
    target_dir (str): 目标目录路径。

    返回:
    None
    """
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)


#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    """按类别比例拆分验证集并组织文件结构。

    参数:
    data_dir (str): 数据根目录路径。
    labels (dict): 文件标识符到标签的映射字典。
    valid_ratio (float): 验证集占原始训练集的比例。

    返回:
    int: 每个类别分配的验证集样本数量。
    """
    # 计算最小类别样本数并确定验证集样本数
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}

    # 遍历训练集文件进行分类复制
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        img_id = train_file.split('.')[0]
        label = labels[img_id]
        fname = os.path.join(data_dir, 'train', train_file)

        # 中间目录统一存放所有训练文件
        copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train_valid', label))

        # 根据计数决定文件最终存放位置
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test', 'train', label))

    return n_valid_per_label


def reorg_test(data_dir):
    """将测试集文件移动到指定目录。

    参数:
    data_dir (str): 数据根目录路径。

    返回:
    None
    """
    test_dir = os.path.join(data_dir, 'test')
    for test_file in os.listdir(test_dir):
        src = os.path.join(test_dir, test_file)
        dst_dir = os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')
        copyfile(src, dst_dir)


def reorg_cifar10_data(data_dir, valid_ratio):
    """统一组织CIFAR-10数据集的文件结构。

    参数:
    data_dir (str): 数据根目录路径。
    valid_ratio (float): 验证集占原始训练集的比例。

    返回:
    None
    """
    labels = read_csv_labels(os.path.join(data_dir, 'trainLabels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)


batch_size = 32 if demo else 128
valid_ratio = 0.1

# 图像增强与标准化变换
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(40),
    torchvision.transforms.RandomResizedCrop(32, scale=(0.64, 1.0), ratio=(1.0, 1.0)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                    [0.2023, 0.1994, 0.2010])
])


# 创建训练/验证/测试数据集
train_ds, train_valid_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_train)
    for folder in ['train', 'train_valid']
]

valid_ds, test_ds = [
    torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=transform_test)
    for folder in ['valid', 'test']
]


# 创建数据加载器
# 训练集和测试集的drop_last参数设置为True，因为要保证每一个batch的大小是相等的。
# 但是验证集的drop_last参数设置为False，不能随意丢弃，这可是要算分的！
train_iter, train_valid_iter = [
    torch.utils.data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    for dataset in (train_ds, train_valid_ds)
]
# 测试数据加载器不需要随机打乱数据，所以此处的shuffle=False
valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size, shuffle=False, drop_last=True)
test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False, drop_last=False)
#

def get_net():
    """创建ResNet-18分类模型。

    返回:
    torch.nn.Module: 初始化的ResNet-18模型。
    """
    num_classes = 10
    # 参见resnet
    net = d2l.resnet18(num_classes, 3)
    return net


loss = nn.CrossEntropyLoss(reduction="none")


def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay):
    """训练模型并监控验证集表现。

    参数:
    net (torch.nn.Module): 目标神经网络模型。
    train_iter (DataLoader): 训练数据迭代器。
    valid_iter (DataLoader): 验证数据迭代器。
    num_epochs (int): 训练轮数。
    lr (float): 初始学习率。
    wd (float): 权重衰减系数。
    devices (list): 使用的GPU设备列表。
    lr_period (int): 学习率调整周期。
    lr_decay (float): 学习率衰减因子。

    返回:
    None
    """
    trainer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                           legend=['train loss', 'train acc', 'valid acc'])

    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    for epoch in range(num_epochs):
        # 训练阶段
        net.train()
        metric = d2l.Accumulator(3)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = d2l.train_batch_ch13(net, features, labels, loss, trainer, devices)
            metric.add(l, acc, labels.shape[0])
            timer.stop()

            # 每五分之一epoch更新可视化
            if (i + 1) % (len(train_iter) // 5) == 0 or i == len(train_iter)-1:
                animator.add(
                    epoch + (i+1)/len(train_iter),
                    (metric[0]/metric[2], metric[1]/metric[2], None)
                )
                plt.pause(0.1)  # 手动刷新

        # 验证阶段
        if valid_iter is not None:
            valid_acc = d2l.evaluate_accuracy_gpu(net, valid_iter)
            animator.add(epoch+1, (None, None, valid_acc))
            plt.pause(0.1)  # 手动刷新

        scheduler.step()  # 更新学习率

    # 训练结束时显示统计信息
    measures = (f'train loss {metric[0]/metric[2]:.3f}, '
                f'train acc {metric[1]/metric[2]:.3f}')
    if valid_iter is not None:
        measures += f', valid acc {valid_acc:.3f}'
    print(f'{measures}\n{metric[2] * num_epochs / timer.sum():.1f} '
          f'examples/sec on {str(devices)}')


# 训练配置参数
devices = d2l.try_all_gpus()
num_epochs, lr, wd = 20, 2e-4, 5e-4
lr_period, lr_decay = 4, 0.9
net = get_net()

# 执行训练流程
# train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 全量训练
net = get_net()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period, lr_decay)

# 生成预测结果
preds = []
for X, _ in test_iter:
    y_hat = net(X.to(devices[0]))
    preds.extend(y_hat.argmax(dim=1).type(torch.int32).cpu().numpy())

# 准备提交文件
sorted_ids = list(range(1, len(test_ds)+1))
sorted_ids.sort(key=lambda x: str(x))
df = pd.DataFrame({'id': sorted_ids, 'label': preds})
df['label'] = df['label'].apply(lambda x: train_valid_ds.classes[x])
df.to_csv('submission.csv', index=False)
