import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from torch import nn
import net


def predict(model, x):
    with torch.no_grad():
        model.eval()
        y = model(x)
        y = f.softmax(y, dim=1)
        model.train()
        return torch.argmax(y, dim=1)


def accuracy(y_hat, y):
    return (y_hat - y).eq(0).float().mean()


def get_kfold_data(k, i, X, y):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid


# 1. Load EEG Data
torch.set_default_dtype(torch.float64)

Data = np.array(pd.read_excel('EEG Data.xlsx', 0, header=None))
for i in range(4):
    nextFrame = np.array(pd.read_excel('EEG Data.xlsx', i+1, header=None))
    Data = np.vstack((Data, nextFrame))  # or np.concatenate((Data,nextFrame),axis=1)

# 2.Shuffle the Data
np.random.shuffle(Data)
Data = torch.from_numpy(Data)


# 3.Get Training and Test Sample/Label form Data
# Training_set = Data[0:2700, :]
# Test_set = Data[2700:3000, :]

# X_train, Y_train = Training_set[:, 1:5], (Training_set[:, 0].t() - 2).long()
# X_test, Y_test = Test_set[:, 1:5], (Test_set[:, 0].t() - 2).long()

X, Y = Data[:, 1:5], (Data[:, 0].t() - 2).long()

# 4.Build model
model = net.ANNmodel()
try:
    model.load_state_dict(torch.load("checkPoint/cp1.pth"))
    print("Check Point Loaded  :)")
except FileNotFoundError as r:
    print("Check Point doesn't exist  :(")


loss_fn = nn.CrossEntropyLoss()  # 这一层包含了softmax+交叉熵损失
lr = 1e-3
epoch = 1000
batch_size = 100
fold = 4

optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # SGD optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) # define Adam optimizer
lossRecord_train = []
lossRecord_test = []
accuracyRecord = []

# 5.Train model
model.train()
for k in range(fold):
    X_train, Y_train, X_test, Y_test = get_kfold_data(fold, k, X, Y)
    for i in range(epoch):
        for j in range(X_train.size(0)//batch_size):  # 批量梯度下降
            y_pred = model(X_train[j*batch_size:(j+1)*batch_size, :])
            loss = loss_fn(y_pred, Y_train[j*batch_size:(j+1)*batch_size])

            # we can also do update by using optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  # update weights

        # with torch.no_grad():  #记录损失(每个epoch
        #     loss_test = loss_fn(model(X_test), Y_test)
        #     loss_train = loss_fn(model(X_train), Y_train)
        #     lossRecord_test.append(loss_test.item())
        #     lossRecord_train.append(loss_train.item())
        if i % 100 == 0:

            y_hat_train = predict(model, X_train)  # y_pred经过softmax后才是真正的输出
            y_hat_test = predict(model, X_test)  # y_pred经过softmax后才是真正的输出
            print(F"fold:{k} | epoch:{i}/{epoch} | Loss:{loss.item()} | Accuracy_train:{accuracy(y_hat_train, Y_train)} | Accuracy_test:{accuracy(y_hat_test, Y_test)}")
    y_hat_test = predict(model, X_test)  # y_pred经过softmax后才是真正的输出
    accuracyRecord.append(accuracy(y_hat_test, Y_test))

# 6.Save Model
torch.save(model.state_dict(), "checkPoint/cp1.pth")
print(F"Mean Accuracy of {fold} Folds: {np.mean(accuracyRecord)}")

plt.plot(lossRecord_train, color='b')
plt.plot(lossRecord_test, color='r')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title('Cross Entropy损失', fontsize=24)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Cross Entropy Value', fontsize=14)
plt.show()

print(Y_train[0:20])
print(predict(model, X_train)[0:20])

