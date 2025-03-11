import torch
import torchvision
import matplotlib.pyplot as plt
import time
import sys

from torch.utils import data
from torchvision import datasets, transforms
from d2l import torch as d2l

d2l.use_svg_display()

mnist_train = torchvision.datasets.FashionMNIST(root="data", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
mnist_test = torchvision.datasets.FashionMNIST(root="data", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)

feature, label = mnist_train[0]

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)