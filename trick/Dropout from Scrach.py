import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # 在本情况中，所有元素都被丢弃
    if dropout == 1:
        return torch.zeros_like(X)
    # 在本情况中，所有元素都被保留
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)

class Net(nn.Module):
    """
    我们可以将暂退法应用于每个隐藏层的输出（在激活函数之后）,并且可以为每一层分别设置暂退概率.
    常见的技巧是在靠近输入层的地方设置较低的暂退概率。 下面的模型将第一个和第二个隐藏层的暂退概率分别设置为0.2和0.5.
    并且暂退法只在训练期间有效!
    """
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                 is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        # 只有在训练模型时才使用dropout
        if self.training == True:
            # 在第一个全连接层之后添加一个dropout层
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            # 在第二个全连接层之后添加一个dropout层
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out

if __name__ == '__main__':
    # 我们定义具有两个隐藏层的多层感知机，每个隐藏层包含256个单元。
    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    dropout1, dropout2 = 0.2, 0.5
    num_epochs, lr, batch_size = 10, 0.5, 256

    net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()