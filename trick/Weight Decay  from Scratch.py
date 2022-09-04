import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]


def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2


def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                            xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加了L2范数惩罚项，
            # 广播机制使l2_penalty(w)成为一个长度为batch_size的向量
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数是：', torch.norm(w).item())


if __name__ == '__main__':
    n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
    true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
    train_data = d2l.synthetic_data(true_w, true_b, n_train)
    train_iter = d2l.load_array(train_data, batch_size)
    test_data = d2l.synthetic_data(true_w, true_b, n_test)
    test_iter = d2l.load_array(test_data, batch_size, is_train=False)

    # 我们现在⽤lambd = 0禁⽤权重衰减后运⾏这个代码。注意，这⾥训练误差有了减少，但测试误差没有减少，
    # 这意味着出现了严重的过拟合。
    train(lambd=0)
    # 使⽤权重衰减来运⾏代码,在这⾥训练误差增⼤，但测试误差减⼩。
    train(lambd=3)
    plt.show()

