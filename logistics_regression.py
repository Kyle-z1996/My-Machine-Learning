import torch
from torch import nn
from torch import mm
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

class LogisticModel(object):
    def __init__(self, x_sample, y_sample, learning_rate):

        self.X = x_sample
        self.Y = y_sample
        self.lr = learning_rate

    def fit(self):
        epoch = 10000
        weights = torch.rand(1, X.shape[1])  # 一次项参数初始化
        One = torch.ones(n, 1)  # 常数项初始化
        bias = 0
        loss = []
        y_predict = [[]]
        for i in range(epoch):
            Z = X.mm(weights.t()) + bias*One
            y_predict = torch.sigmoid(Z)
            weights -= self.grad_w(y_predict, Y, X)*self.lr
            bias -= self.grad_b(y_predict, Y)*self.lr

            if(i % 100 == 0):
                loss.append(self.loss_func(y_predict, Y))
                print(F"epoch {i} | loss: {loss[-1]} | accuracy:{accuracy(y_predict.gt(0.5), Y)}")

        return weights, bias, loss, y_predict

    def loss_func(self, y_pred, y):
        return -(y*torch.log(y_pred) + (1-y)*torch.log(1-y_pred)).mean()

    def grad_w(self, y_pred, y, X):
        return (y_pred - y).t().mm(X) / y.size(0)  # y:n x 1  w:1 x 2

    def grad_b(self, y_pred, y):
        return (y_pred - y).mean()


def accuracy(y_pred, y):
    return y_pred.bool().eq(y.bool()).float().mean()

torch.set_default_dtype(torch.float64)

n = 100

X = 2*torch.randn(n, 2)  # 生成n个样本 (高斯分布)
W = torch.tensor([[0.4, 0.6]], dtype=torch.float64)  # 二元线性
b = 0.4
Y = torch.sigmoid(X.mm(W.t()) + b * torch.ones(n, 1) + 0.1*torch.randn(n, 1))  # 生成带噪声样本 Y = XW + b + noise(高斯噪声)
Y = Y.gt(Y.mean()).float()  # 结果为0/1

lm = LogisticModel(X, Y, 5e-3)

W_fit, b_fit, loss, Y_pred = lm.fit()

print("测试值:")
print(W, b)
print("拟合值:")
print(W_fit, b_fit.item())
print(F"正确率:{accuracy(Y_pred.gt(0.5), Y)}")
plt.plot(loss)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Cross Entropy损失', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Cross Entropy Value', fontsize=14)
plt.show()