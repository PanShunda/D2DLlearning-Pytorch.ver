import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 加载时间机器数据集，创建数据迭代器和词汇表
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

num_hiddens = 256
# 初始化RNN层，输入大小为词汇表大小，隐藏单元数为256
rnn_layer = nn.RNN(len(vocab), num_hiddens)

# 初始化隐藏状态，形状为(层数, 批量大小, 隐藏单元数)
state = torch.zeros((1, batch_size, num_hiddens))

# 生成随机输入张量并进行前向传播测试
X = torch.rand(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)

class RNNModel(nn.Module):
    """循环神经网络模型。"""
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        """初始化循环神经网络模型。
        Args:
            rnn_layer (nn.Module): 循环神经网络层实例（如nn.RNN）
            vocab_size (int): 词汇表的大小
            **kwargs: 其他关键字参数
        """
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        """前向传播计算模型输出。
        Args:
            inputs (Tensor): 输入序列，形状为(时间步数, 批量大小)
            state (Tensor或元组): 循环层的隐藏状态
        Returns:
            output (Tensor): 模型预测的输出，形状为(时间步数*批量大小, 词汇表大小)
            state (Tensor或元组): 更新后的隐藏状态
        """
        # 将输入转置并转换为one-hot编码，调整为浮点类型
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        # 通过RNN层进行前向传播，得到输出和新的隐藏状态
        Y, state = self.rnn(X, state)
        # 线性层处理输出，调整形状以便预测
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        """返回模型的初始隐藏状态。
        Args:
            device (torch.device): 设备（CPU或GPU）
            batch_size (int): 批量大小，默认为1
        Returns:
            Tensor或元组: 初始隐藏状态（如果是LSTM则返回元组，否则是单个Tensor）
        """
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens), device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers,
                                 batch_size, self.num_hiddens),
                                device=device))

# 将模型移动到GPU并进行预测
device = d2l.try_gpu()
net = RNNModel(rnn_layer, vocab_size=len(vocab))
net = net.to(device)
d2l.predict_ch8('time traveller', 10, net, vocab, device)
