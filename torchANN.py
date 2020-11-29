import torch
from torch import nn
from torch import mm
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

def predict(model, x):
    with torch.no_grad():
        y = model(x)
        Y = F.softmax(y, dim=1)
        return torch.argmax(Y, dim=1)

def accuracy(y_pred, y):
    return (y_pred - y).eq(0).float().mean()


model = torch.nn.Sequential(
    torch.nn.Linear(2, 5),
    torch.nn.Tanh(),
    torch.nn.Linear(5, 2),
)

loss_fn = nn.CrossEntropyLoss()  # 这一层包含了softmax+交叉熵损失
lr = 1e-2
# 样本生成
np.random.seed(0)
X, y = sklearn.datasets.make_moons(500, noise=0.2)
X = 10*torch.from_numpy(X).float()
Y = torch.tensor(y).long()

X_train, Y_train = X[0:300, :], Y[0:300]  # training set
X_validation, Y_validation = X[300:400, :], Y[300:400]  # validation set
X_test, Y_test = X[400:500, :], Y[400:500]  # test set

epoch = 2000
batch_size = 50

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # define SGD optimizer

lossRecord_train = []
lossRecord_test = []

for i in range(epoch):
    for j in range(X_train.size(0)//batch_size):  # 批量梯度下降
        y_pred = model(X_train[j*batch_size:(j+1)*batch_size, :])
        loss = loss_fn(y_pred, Y_train[j*batch_size:(j+1)*batch_size])

        # update weights manually
        # model.zero_grad()
        # loss.backward()  # back propagation
        # with torch.no_grad():  # update weights
        #     for param in model.parameters():
        #         param -= lr * param.grad

        # we can also do update by using optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # update weights


    with torch.no_grad():  #记录损失
        loss_test = loss_fn(model(X_test), Y_test)
        loss_train = loss_fn(model(X_train), Y_train)
        lossRecord_test.append(loss_test.item())
        lossRecord_train.append(loss_train.item())
    if i % 100 == 0:
        with torch.no_grad():
            y_hat_train = predict(model, X_train)  # y_pred经过softmax后才是真正的输出
            y_hat_test = predict(model, X_test)  # y_pred经过softmax后才是真正的输出
            print(F"epoch:{i}/{epoch} | Loss:{loss.item()} | Accurary_train:{accuracy(y_hat_train, Y_train)} | Accurary_test:{accuracy(y_hat_test, Y_test)}")





plt.plot(lossRecord_train, color='b')
plt.plot(lossRecord_test, color='r')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Cross Entropy损失', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Cross Entropy Value', fontsize=14)
plt.show()
