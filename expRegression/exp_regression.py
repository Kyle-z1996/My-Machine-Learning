import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ExpModel(object):
    def __init__(self, x_sample, y_sample, learning_rate, reg_lambda):
        # x_sample: n x 2
        # y_sample: n维向量
        np.random.seed(0)
        self.X = x_sample
        self.Y = y_sample.reshape(len(y_sample), 1)  # 转换为 nx1 矩阵

        # 初始化参数
        # 回归目标: y = exp(w1*x1 + b1) + w2*x2 + b2
        self.parameters = np.array([0.52, -4.651, 0.285, 168.217])  # [w1, b1, w2, b2]
        # self.parameters = np.array([0.684, -7.679, 0.51, 166.49])  # [w1, b1, w2, b2]
        # self.parameters = np.random.randn(4)  # [w1, b1, w2, b2]

        # 梯度记录
        self.grad = np.array([0, 0, 0, 0])  # [dw1, db1, dw2, db2]

        # 学习率与正则化参数
        self.lr = learning_rate
        self.reg_lambda = reg_lambda

    def fit(self):
        epoch = 20000
        batch_size = 40
        lossRecord = []

        for i in range(epoch):
            # 随机打乱样本
            self.X, self.Y = shuffle(self.X, self.Y)
            for j in range(self.X.shape[0] // batch_size):  # SGD
                x_batch = self.X[j * batch_size:(j + 1) * batch_size, :]
                y_batch = self.Y[j * batch_size:(j + 1) * batch_size]

                # 生成梯度
                self.forward(x_batch, y_batch)
                self.parameters -= self.lr*self.grad

            if i % 1000 == 0:
                y_predict = self.predict(self.X)
                loss = self.mseLoss(y_predict, self.Y)
                lossRecord.append(loss)
                print(F"epoch {i} | loss: {lossRecord[-1]}")

        return self.parameters

    def mseLoss(self, y_pred, y):
        return np.square(y_pred - y).mean()


    def predict(self, x):
        w1, b1, w2, b2 = self.parameters[0], self.parameters[1], self.parameters[2], self.parameters[3]
        # 是否考虑x2
        y = np.exp(w1*x[:, 0] + b1) + w2*x[:, 1] + b2
        # y = np.exp(w1 * x[:, 0] + b1) + b2
        return y.reshape(len(y), 1)

    def forward(self, x, y):
        w1, b1 = self.parameters[0], self.parameters[1]
        x1 = x[:, 0]
        x2 = x[:, 1]
        # 样本数量: n
        n = x1.shape[0]
        expOut = np.exp(w1*x[:, 0] + b1)
        y_pred = self.predict(x)
        dy = (y_pred - y)  # nx1
        dexp = expOut  # nx1

        # clipping 防止指数函数梯度爆炸/消失
        dexp = np.clip(dexp, 1e-1, 20)
        dy = np.clip(dy, -20, 20)

        # 链式法则
        dw1 = np.dot((dy*dexp).T, x1).mean()/n + self.reg_lambda*self.grad[0]
        db1 = (dy*dexp).mean()
        dw2 = np.dot(dy.T, x2).mean()/n + self.reg_lambda*self.grad[2]
        db2 = dy.mean()

        # 动量梯度下降
        # beta = 0.5
        # self.grad = np.array([dw1, db1, dw2, db2])*beta + self.grad*(1 - beta)  # 记录梯度

        self.grad = np.array([dw1, db1, dw2, db2])  # 记录梯度


def shuffle(X, Y):
    # print(X)
    # print(Y)
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    shuffled_x = data[:, 0:2]
    shuffled_y = data[:, 2]
    shuffled_y = shuffled_y.reshape(len(shuffled_y), 1)
    return shuffled_x, shuffled_y


if __name__ == "__main__":
    df = pd.read_excel("delay_DataSet.xlsx", header=0)
    X = np.array(df)[:, 0:2]  # nx2矩阵
    Y = np.array(df)[:, 2]  # n维向量
    # X:输入 Y:输出 lr:学习率 lambda:正则化参数
    model = ExpModel(X, Y, learning_rate=1e-7, reg_lambda=0)
    # 使用梯度下降进行拟合
    parameters = model.fit()
    Y_pred = model.predict(X)
    # 拟合标准差
    std = np.sqrt(model.mseLoss(Y_pred, Y.reshape(len(Y), 1)))

    plt.plot(Y, label='mean_delay', color='b')
    plt.plot(Y_pred, label='predict', color='orange')
    plt.rcParams['font.sans-serif'] = ['SimHei']
    w1, b1, w2, b2 = parameters[0], parameters[1], parameters[2], parameters[3]
    print(F"\nmean_delay = exp({round(w1,3)}*[EMA_requests] + {round(b1,3)}) + {round(w2,3)}*[requests] + {round(b2,3)} ± {std}")
    plt.title('订单延迟预测', fontsize=14)
    plt.xlabel('时间', fontsize=14)
    plt.ylabel('延迟/s', fontsize=14)
    plt.show()



