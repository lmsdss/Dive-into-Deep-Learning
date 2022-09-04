import math
import torch
from torch import nn
from d2l import torch as d2l

# 由于查询、键和值来自同一组输入，因此被称为自注意力（self-attention）。
# 自注意力同时具有并行计算和最短的最大路径长度这两个优势。

# 为了使用序列的顺序信息，我们通过在输入表示中添加位置编码（positional encoding）来注入绝对的或相对的位置信息。
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == '__main__':

    # 下面的代码片段是基于多头注意力对一个张量完成自注意力的计算。
    # 张量的形状为（批量大小，时间步的数目或词元序列的长度，d）。
    # 输出与输入的张量形状相同。
    num_hiddens, num_heads = 100, 5
    attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                       num_hiddens, num_heads, 0.5)
    attention.eval()

    batch_size, num_queries, valid_lens = 2, 4, torch.tensor([3, 2])
    X = torch.ones((batch_size, num_queries, num_hiddens))
    print(attention(X, X, X, valid_lens).shape)

    encoding_dim, num_steps = 32, 60
    pos_encoding = PositionalEncoding(encoding_dim, 0)
    pos_encoding.eval()
    X = pos_encoding(torch.zeros((1, num_steps, encoding_dim)))
    P = pos_encoding.P[:, :X.shape[1], :]
    d2l.plot(torch.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
             figsize=(6, 2.5), legend=["Col %d" % d for d in torch.arange(6, 10)])