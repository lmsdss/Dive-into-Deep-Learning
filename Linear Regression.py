import torch
from torch.utils import data
from d2l import torch as d2l
from torch import nn

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    # 布尔值is_train表示是否希望数据迭代器对象在每个迭代周期内打乱数据。
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


batch_size = 10
data_iter = load_array((features, labels), batch_size)

# print(next(iter(data_iter)))  # 使⽤iter构造Python迭代器，并使⽤next从迭代器中获取第⼀项。

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
# 通过net[0]选择⽹络中的第⼀个图层，然后使⽤weight.data和bias.data⽅法访问参数。
net[0].weight.data.normal_(0, 0.01) # normal_(0, 0.01)用均值为0、标准差为0.01的正态分布
net[0].bias.data.fill_(0) # fill_(0)就表示用0填充

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

if __name__ == '__main__':
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()  # 通过进⾏反向传播来计算梯度。
            trainer.step()  # 通过调⽤优化器来更新模型参数。
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)
