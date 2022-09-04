import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt

# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


net.apply(init_weights)

loss = nn.CrossEntropyLoss(reduction='none')

trainer = torch.optim.SGD(net.parameters(), lr=0.1)

if __name__ == '__main__':
    lr = 0.1
    num_epochs = 10
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    plt.show()
