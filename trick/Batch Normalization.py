import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

# 批量规范化层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。
net = nn.Sequential(
    # 批量规范化是在卷积层或全连接层之后、相应的激活函数之前应用的。
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))

if __name__ == '__main__':
    lr, num_epochs, batch_size = 1.0, 10, 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()