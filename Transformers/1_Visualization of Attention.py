import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt



def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(4, 2.5), cmap='Reds'):
    """为了可视化注意力权重，我们定义了show_heatmaps函数。
      其输入matrices的形状是 （要显示的行数，要显示的列数，查询的数目，键的数目）。"""
    """显示矩阵热图"""
    d2l.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)
    plt.show()


if __name__ == '__main__':
    # 本例子中，仅当查询和键相同时，注意力权重为1，否则为0。
    attention_weights = torch.eye(15).reshape((1, 1, 15, 15))
    show_heatmaps(attention_weights, xlabel='Keys', ylabel='Queries')
