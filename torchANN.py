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
        Y = nn.functional.softmax(y, dim=1).numpy()
    return torch.from_numpy(np.argmax(Y, axis=1)).float()

def accuracy(y_pred, y):
    return (y_pred - y).eq(0).float().mean()


model = torch.nn.Sequential(
    torch.nn.Linear(2, 5),
    torch.nn.Tanh(),
    torch.nn.Linear(5, 2),
)

loss_fn = nn.CrossEntropyLoss()  # 这一层包含了softmax+交叉熵损失
lr = 5e-3
# 样本生成
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.2)
X = 10*torch.from_numpy(X).float()
Y = torch.tensor(y).long()
lossRecord = []

for t in range(20000):
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    lossRecord.append(loss.item())
    if t % 1000 == 1:
        with torch.no_grad():
            y_hat = predict(model, X)  # y_pred经过softmax后才是真正的输出
            print(F"epoch:{t} | Loss:{loss.item()} | Accurary:{accuracy(y_hat, y)}")
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad


plt.plot(lossRecord)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Cross Entropy损失', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Cross Entropy Value', fontsize=14)
plt.show()
