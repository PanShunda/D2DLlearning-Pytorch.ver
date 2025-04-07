import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

F.one_hot(torch.tensor([0, 2]), len(vocab))

X = torch.arange(10).reshape((2, 5))

def get_params(vocab_size, num_hiddens, device):
    """初始化RNN模型参数
    参数:
        vocab_size (int): 词汇表大小
        num_hiddens (int): 隐藏层单元数量
        device (torch.device): 运行设备
    返回:
        list: 包含以下参数的列表:
            W_xh (Tensor): 输入到隐藏层的权重矩阵
            W_hh (Tensor): 隐藏层自连接权重矩阵
            b_h (Tensor): 隐藏层偏置
            W_hq (Tensor): 隐藏层到输出层的权重矩阵
            b_q (Tensor): 输出层偏置
    """
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

def init_rnn_state(batch_size, num_hiddens, device):
    """初始化RNN隐藏状态
    参数:
        batch_size (int): 批量大小
        num_hiddens (int): 隐藏层单元数量
        device (torch.device): 运行设备
    返回:
        tuple: 包含形状为(batch_size, num_hiddens)的零张量的元组
    """
    return (torch.zeros((batch_size, num_hiddens), device=device),)

def rnn(inputs, state, params):
    """RNN模型前向计算函数
    参数:
        inputs (Tensor): 输入序列张量，形状为(num_steps, batch_size, vocab_size)
        state (tuple): 当前隐藏状态
        params (list): 模型参数列表
    返回:
        tuple: 输出张量（形状为(num_steps*batch_size, vocab_size)）和新隐藏状态
    """
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device, get_params,
                 init_state, forward_fn):
        """初始化模型参数和函数
        参数:
            vocab_size (int): 词汇表大小
            num_hiddens (int): 隐藏层单元数量
            device (torch.device): 运行设备
            get_params (function): 参数初始化函数
            init_state (function): 状态初始化函数
            forward_fn (function): 前向传播函数
        """
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        """模型前向计算
        参数:
            X (Tensor): 输入序列，形状为(num_steps, batch_size)
            state (tuple): 当前隐藏状态
        返回:
            tuple: 输出张量和新隐藏状态
        """
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        """获取初始隐藏状态
        参数:
            batch_size (int): 批量大小
            device (torch.device): 运行设备
        返回:
            tuple: 初始隐藏状态
        """
        return self.init_state(batch_size, self.num_hiddens, device)

num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)

def predict_ch8(prefix, num_preds, net, vocab, device):
    """在`prefix`后面生成新字符
    参数:
        prefix (str): 起始字符串
        num_preds (int): 需要生成的字符数量
        net: RNN模型
        vocab: 词汇表对象
        device (torch.device): 运行设备
    返回:
        str: 包含起始和生成字符的完整字符串
    """
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape(
        (1, 1))
    for y in prefix[1:]:
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])

predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())

def grad_clipping(net, theta):
    """裁剪梯度
    参数:
        net: 模型对象（可以是nn.Module或自定义模型）
        theta (float): 梯度裁剪阈值
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练模型一个迭代周期
    参数:
        net: 模型对象
        train_iter: 数据迭代器
        loss: 损失函数
        updater: 优化器或更新函数
        device (torch.device): 运行设备
        use_random_iter (bool): 是否使用随机采样
    返回:
        tuple: (困惑度, 每秒处理的标记数)
    """
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state is None or use_random_iter:
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    """训练模型并监控训练过程
    参数:
        net: 模型对象
        train_iter: 数据迭代器
        vocab: 词汇表对象
        lr (float): 学习率
        num_epochs (int): 训练轮数
        device (torch.device): 运行设备
        use_random_iter (bool): 是否使用随机采样
    """
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device,
                                     use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 标记/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
