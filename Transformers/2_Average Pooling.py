import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def f(x):
    return 2 * torch.sin(x) + x ** 0.8


def plot_kernel_reg(y_hat):
    d2l.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()


class NWKernelRegression(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = nn.Parameter(torch.rand((1,), requires_grad=True))

    def forward(self, queries, keys, values):
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1)
        # values的形状为(查询个数，“键－值”对个数)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)


if __name__ == '__main__':
    n_train = 50  # 训练样本数
    x_train, _ = torch.sort(torch.rand(n_train) * 5)  # 排序后的训练样本

    y_train = f(x_train) + torch.normal(0.0, 0.5, (n_train,))  # 训练样本的输出
    x_test = torch.arange(0, 5, 0.1)  # 测试样本
    y_truth = f(x_test)  # 测试样本的真实输出
    n_test = len(x_test)  # 测试样本数

    # 1 平均汇聚:  基于平均汇聚来计算所有训练样本输出值的平均值,但平均汇聚忽略了输入xi。
    y_hat = torch.repeat_interleave(y_train.mean(), n_test)
    plot_kernel_reg(y_hat)

    # 2 非参数的注意力汇聚:  Nadaraya-Watson核回归 根据输入的位置对输出yi进行加权
    # X_repeat的形状:(n_test,n_train),
    # 每一行都包含着相同的测试输入（例如：同样的查询）
    X_repeat = x_test.repeat_interleave(n_train).reshape((-1, n_train))
    # x_train包含着键。attention_weights的形状：(n_test,n_train),
    # 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
    attention_weights = nn.functional.softmax(-(X_repeat - x_train) ** 2 / 2, dim=1)
    # y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
    y_hat = torch.matmul(attention_weights, y_train)
    plot_kernel_reg(y_hat)

    """
    我们来观察注意力的权重。 这里测试数据的输入相当于查询，而训练数据的输入相当于键。 
    因为两个输入都是经过排序的，因此由观察可知“查询-键”对越接近，注意力汇聚的注意力权重就越高。
    """
    d2l.show_heatmaps(attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')


    # 3 带参数注意力汇聚:查询x和键xi之间的距离乘以可学习参数w
    # 将训练数据集变换为键和值用于训练注意力模型。
    # 任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算， 从而得到其对应的预测输出。
    # X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
    X_tile = x_train.repeat((n_train, 1))
    # Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
    Y_tile = y_train.repeat((n_train, 1))
    # keys的形状:('n_train'，'n_train'-1)
    keys = X_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values的形状:('n_train'，'n_train'-1)
    values = Y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))

    net = NWKernelRegression()
    loss = nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
        animator.add(epoch + 1, float(l.sum()))

    # keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
    keys = x_train.repeat((n_test, 1))
    # value的形状:(n_test，n_train)
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    # 在尝试拟合带噪声的训练数据时， 预测结果绘制的线不如之前非参数模型的平滑。
    plot_kernel_reg(y_hat)

    # 与非参数的注意力汇聚模型相比， 带参数的模型加入可学习的参数后， 曲线在注意力权重较大的区域变得更不平滑。
    d2l.show_heatmaps(net.attention_weights.unsqueeze(0).unsqueeze(0),
                      xlabel='Sorted training inputs',
                      ylabel='Sorted testing inputs')



