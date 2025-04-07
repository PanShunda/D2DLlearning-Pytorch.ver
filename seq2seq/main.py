import collections
import math
import torch
from torch import nn
from d2l import torch as d2l

class Seq2SeqEncoder(d2l.Encoder):
    """用于序列到序列学习的循环神经网络编码器。

    参数：
    vocab_size (int): 源语言词汇表大小
    embed_size (int): 嵌入层输出维度
    num_hiddens (int): 隐藏层神经元数量
    num_layers (int): RNN网络层数
    dropout (float): 丢弃层概率
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, *args):
        """前向计算

        参数：
        X (Tensor): 输入序列 (batch_size, seq_len)

        返回：
        output (Tensor): RNN输出序列 (seq_len, batch_size, num_hiddens)
        state (Tensor): 隐藏状态 (num_layers, batch_size, num_hiddens)
        """
        X = self.embedding(X)
        X = X.permute(1, 0, 2)  # 转换为 (seq_len, batch_size, embed_size)
        output, state = self.rnn(X)
        return output, state

class Seq2SeqDecoder(d2l.Decoder):
    """用于序列到序列学习的循环神经网络解码器。

    参数：
    vocab_size (int): 目标语言词汇表大小
    embed_size (int): 嵌入层输出维度
    num_hiddens (int): 隐藏层神经元数量
    num_layers (int): RNN网络层数
    dropout (float): 丢弃层概率
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, *args):
        """初始化解码器状态

        参数：
        enc_outputs (tuple): 编码器输出 (outputs, states)

        返回：
        state (Tensor): 初始隐藏状态 (num_layers, batch_size, num_hiddens)
        """
        return enc_outputs[1]

    def forward(self, X, state):
        """前向计算

        参数：
        X (Tensor): 输入序列 (batch_size, seq_len)
        state (Tensor): 隐藏状态 (num_layers, batch_size, num_hiddens)

        返回：
        output (Tensor): 全连接层输出 (batch_size, vocab_size)
        state (Tensor): 更新后的隐藏状态
        """
        X = self.embedding(X).permute(1, 0, 2)  # 转换为 (seq_len, batch_size, embed_size)
        context = state[-1].repeat(X.shape[0], 1, 1)  # 复制上下文向量到每个时间步
        X_and_context = torch.cat((X, context), 2)  # 拼接输入和上下文
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)  # 转换为 (batch_size, seq_len, vocab_size)
        return output, state

def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项。

    参数：
    X (Tensor): 输入张量 (batch_size, seq_len)
    valid_len (Tensor): 每个序列的有效长度 (batch_size,)
    value (float): 填充值

    返回：
    Tensor: 掩码后的张量
    """
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    def forward(self, pred, label, valid_len):
        """计算遮蔽后的损失

        参数：
        pred (Tensor): 预测值 (batch_size, seq_len, vocab_size)
        label (Tensor): 标签 (batch_size, seq_len)
        valid_len (Tensor): 有效长度 (batch_size,)

        返回：
        Tensor: 每个样本的平均损失
        """
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss,
                                self).forward(pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型。

    参数：
    net (nn.Module): 序列到序列模型
    data_iter (DataLoader): 数据迭代器
    lr (float): 学习率
    num_epochs (int): 训练轮数
    tgt_vocab (Vocab): 目标语言词汇表
    device (torch.device): 计算设备
    """
    def xavier_init_weights(m):
        """Xavier权重初始化"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # 记录总损失和词数
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测

    参数：
    src_sentence (str): 源语言句子
    src_vocab (Vocab): 源语言词汇表
    tgt_vocab (Vocab): 目标语言词汇表
    num_steps (int): 最大生成步数
    device (torch.device): 计算设备
    save_attention_weights (bool): 是否保存注意力权重

    返回：
    str: 生成的目标语言句子
    list: 注意力权重序列（可选）
    """
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long, device=device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    dec_X = torch.unsqueeze(
        torch.tensor([tgt_vocab['<bos>']], dtype=torch.long, device=device),
        dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq

def bleu(pred_seq, label_seq, k):
    """计算 BLEU

    参数：
    pred_seq (str): 预测序列
    label_seq (str): 标签序列
    k (int): n-gram最大长度

    返回：
    float: BLEU分数
    """
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[''.join(label_tokens[i:i + n])] += 1
        for i in range(len_pred - n + 1):
            gram = ''.join(pred_tokens[i:i + n])
            if label_subs[gram] > 0:
                num_matches += 1
                label_subs[gram] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
