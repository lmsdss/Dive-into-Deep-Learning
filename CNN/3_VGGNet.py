import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt


def vgg_block(num_convs, in_channels, out_channels):
    # 卷积层的数量num_convs、输入通道的数量in_channels 和输出通道的数量out_channels.
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*layers)


# 该变量指定了每个VGG块里卷积层个数和输出通道数。
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 代码实现了VGG-11。可以通过在conv_arch上执行for循环来简单实现。
def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))


net = vgg(conv_arch)

# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

# 由于VGG-11比AlexNet计算量更大，因此可以构建了一个通道数较少的网络，足够用于训练Fashion-MNIST数据集。
ratio = 4
# [(1, 16), (1, 32), (2, 64), (2, 128), (2, 128)]
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net_small= vgg(small_conv_arch)

if __name__ == '__main__':
    lr, num_epochs, batch_size = 0.05, 10, 128
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    d2l.train_ch6(net_small, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
    plt.show()
