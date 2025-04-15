"""
Bahdanau
加性注意力的seq2seq
"""


import torch
from torch import nn
from d2l import torch as d2l

class AttentionDecoder(d2l.Decoder):
    """带有注意力机制的解码器基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

class Seq2SeqAttentionDecoder(AttentionDecoder):
    """带有注意力机制的序列到序列解码器类。

    参数：
    vocab_size (int): 目标词汇表大小
    embed_size (int): 词嵌入维度
    num_hiddens (int): 隐藏层单元数量
    num_layers (int): RNN层的数量
    dropout (float): dropout比率
    """
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        # 初始化加性注意力层
        self.attention = d2l.AdditiveAttention(num_hiddens, num_hiddens,
                                               num_hiddens, dropout)
        # 初始化词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 初始化GRU网络层（输入合并了上下文向量）
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        # 初始化最终输出全连接层
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        """初始化解码器状态。

        参数：
        enc_outputs (tuple): 编码器输出（包含输出序列和隐藏状态）
        enc_valid_lens (Tensor): 编码器有效长度
        *args: 其他可变参数

        返回：
        tuple: 重组后的解码器初始状态（编码器输出、隐藏状态、有效长度）
        """
        outputs, hidden_state = enc_outputs
        return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        """执行前向计算的解码过程。

        参数：
        X (Tensor): 输入序列（形状：batch_size，时间步数）
        state (tuple): 解码器状态（编码器输出、隐藏状态、有效长度）

        返回：
        tuple: 解码器输出（形状：batch_size，时间步数，vocab_size）和新状态
        """
        enc_outputs, hidden_state, enc_valid_lens = state
        # 进行词嵌入并调整张量维度
        X = self.embedding(X).permute(1, 0, 2)
        outputs, self._attention_weights = [], []
        for x in X:
            # 提取最后一层隐藏状态作为查询向量
            query = torch.unsqueeze(hidden_state[-1], dim=1)
            # 计算注意力加权的上下文向量
            context = self.attention(query, enc_outputs, enc_outputs,
                                     enc_valid_lens)
            # 合并上下文向量与当前输入
            x = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            # 通过GRU网络处理合并后的输入
            out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
            outputs.append(out)
            self._attention_weights.append(self.attention.attention_weights)
        # 最终输出通过全连接层并调整维度
        outputs = self.dense(torch.cat(outputs, dim=0))
        return outputs.permute(1, 0, 2), [
            enc_outputs, hidden_state, enc_valid_lens]

    @property
    def attention_weights(self):
        """获取注意力权重属性。

        返回：
        list: 包含各时间步注意力权重的列表
        """
        return self._attention_weights





