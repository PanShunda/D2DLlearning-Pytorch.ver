## seq2seq

重点理解一下其中的编码器解码器结构：

在我的理解中，如果只看编码器，它就是一个RNN，但是解码器则有一些不同。

首先是将什么东西输入进了解码器呢？是input和编码器的最后一个时间步的状态拼接而成的向量。

#### 编码器 `Seq2SeqEncoder.forward(X)`

- 输入 `X` 形状是 `(batch_size, seq_len)`
- 经过 `nn.Embedding` 后变为 `(batch_size, seq_len, embed_size)`
- 再通过 `.permute(1, 0, 2)` 变成 `(seq_len, batch_size, embed_size)` → 这是 RNN 的标准输入格式
- 最终传入 `RNN`，输出：
  - `output`: `(seq_len, batch_size, num_hiddens)`
  - `state`: `(num_layers, batch_size, num_hiddens)`

#### 解码器 `Seq2SeqDecoder.forward(X, state)`

- 输入 `X` 形状同样是 `(batch_size, seq_len)`，经过和编码器一样的处理方式进入 `GRU`
- 区别在于：
  - 解码器会**将编码器的最后一层隐藏状态作为上下文向量 context**，与输入拼接
  - 拼接后的维度是 `(seq_len, batch_size, embed_size + num_hiddens)`

GRU 输出的状态是由 `hidden_size` 决定的，和输入特征数无关！

因此编码器输入到RNN的形状是(batch_size, seq_len)

经过embeding之后还要调整一下形状参数位置，输入前的形状为(seq_len, batch_size, embed_size)

输出的state形状是(num_layers, batch_size, num_hiddens)

与X进行拼接时只使用了最后一层的隐藏状态，于是准备拼接的state形状是(batch_size, num_hiddens)再将它复制成每个时间步，于是乎形状便成了(seq_len,batch_size, num_hiddens)

同时X的形状是(seq_len, batch_size, embed_size)

但是会将X与state进行拼接之后再输入进解码器的RNN，形状是(seq_len, batch_size, embed_size + num_hiddens)