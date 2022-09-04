import math
import torch
from torch import nn
from d2l import torch as d2l


# @save
# softmax操作用于输出一个概率分布作为注意力权重。
def masked_softmax(X, valid_lens):
    """我们可以指定一个有效序列长度（即词元的个数）， 以便在计算softmax时过滤掉超出指定范围的位置。"""
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
# # 我们也可以使用二维张量，为矩阵样本中的每一行指定有效长度。
# print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))

# @save
class AdditiveAttention(nn.Module):
    """加性注意力"""

    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)


# @save
class DotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)


if __name__ == '__main__':
    # 1 加性注意力:当查询和键是不同长度的矢量时， 我们可以使用加性注意力作为评分函数。
    """我们用一个小例子来演示上面的AdditiveAttention类， 其中查询、键和值的形状为（批量大小，步数或词元序列长度，特征大小）， 
    实际输出为(2,1,20)、(2,10,2)和(2,10,4)。 注意力汇聚输出的形状为（批量大小，查询的步数，值的维度）。
    """
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.ones((2, 10, 2))
    # values的小批量，两个值矩阵是相同的
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])

    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    attention(queries, keys, values, valid_lens)

    # 尽管加性注意力包含了可学习的参数，但由于本例子中每个键都是相同的， 所以注意力权重是均匀的，由指定的有效长度决定。
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')

    # 2 缩放点积注意力
    # 使用点积可以得到计算效率更高的评分函数， 但是点积操作要求查询和键具有相同的长度d。
    # 为确保无论向量长度如何， 点积的方差在不考虑向量长度的情况下仍然是， 我们将点积除以根号d。
    # 为了演示上述的DotProductAttention类， 我们使用与先前加性注意力例子中相同的键、值和有效长度。
    # 对于点积操作，我们令查询的特征维度与键的特征维度大小相同。
    queries = torch.normal(0, 1, (2, 1, 2))

    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    attention(queries, keys, values, valid_lens)

    # 与加性注意力演示相同，由于键包含的是相同的元素， 而这些元素无法通过任何查询进行区分，因此获得了均匀的注意力权重。
    d2l.show_heatmaps(attention.attention_weights.reshape((1, 1, 2, 10)),
                      xlabel='Keys', ylabel='Queries')

    """
    注意力分数是query和key的相似度，注意力权重是分数softmax的结果。
    将注意力汇聚的输出计算可以作为值的加权平均，选择不同的注意力评分函数会带来不同的注意力汇聚操作。
    当查询和键是不同长度的矢量时，可以使用可加性注意力评分函数。将query和key合并起来进入一个单输出隐藏层的MLP。
    当它们的长度相同时，使用缩放的“点－积”注意力评分函数的计算效率更高。直接将query和key做内积。
    """