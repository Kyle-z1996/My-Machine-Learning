import torch
from torch import nn
from torch import mm
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

class LinearModel(object):
    def __init__(self, x_sample, y_sample, learning_rate):

        self.X = x_sample
        self.Y = y_sample
        self.lr = learning_rate

    def fit(self):
        epoch = 300
        weights = torch.rand(1, 2)  # 一次项参数初始化
        One = torch.ones(n, 1)  # 常数项初始化
        bias = 0
        loss = []

        for i in range(epoch):
            y_predict = X.mm(weights.reshape(2, 1)) + bias*One
            weights -= self.grad_w(y_predict, Y)*self.lr
            bias -= self.grad_b(y_predict, Y)*self.lr

            if(i % 5 == 0):
                loss.append(self.loss_func(y_predict, Y))
                print(F"epoch {i} | loss: {loss[-1]}")

        return weights, bias, loss

    def loss_func(self, y_pred, y):
        return np.square(y_pred - y).mean()

    def grad_w(self, y_pred, y):
        return (y_pred - y).t().mm(X) / y.size(0)  # y:n x 1  w:1 x 2

    def grad_b(self, y_pred, y):
        return (y_pred - y).mean()

n = 100

X = 10*torch.rand(n, 2)  # 生成n个样本 (0-1 均匀分布)
W = torch.tensor([3., 8], dtype=torch.float32)  # 二元线性方程
b = 4

Y = X.mm(W.reshape(2, 1)) + b * torch.ones(n, 1) + torch.randn(n, 1)  # 生成带噪声样本 Y = XW + b + noise(高斯噪声)
lm = LinearModel(X, Y, 2e-3)

W_fit, b_fit, loss = lm.fit()

print("测试值:")
print(W, b)
print("拟合值:")
print(W_fit, b_fit.item())
plt.plot(loss)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('MSE损失', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MES Value', fontsize=14)
plt.show()



