import torch
from torch import nn
from torch import mm
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

model = torch.nn.Sequential(
    torch.nn.Linear(2, 5),
    torch.nn.Tanh(),
    torch.nn.Linear(5, 2),
)

loss_fn = nn.CrossEntropyLoss()
lr = 1e-2

# 样本生成
np.random.seed(0)
X, y = sklearn.datasets.make_moons(100, noise=0.20)
X = torch.from_numpy(X).float()
Y = torch.tensor(y).long()

for t in range(5000):
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)
    if t % 100 == 50:
        print(F"epoch:{t} | Loss:{loss.item()}")
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad



