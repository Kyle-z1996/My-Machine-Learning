import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt


np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
# plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
# plt.show()

num_examples = len(X)  # 训练样本的数量
nn_input_dim = 2  # 输入层的维度
nn_output_dim = 2  # 输出层的维度

# 梯度下降的参数（我直接手动赋值）
epsilon = 0.01  # 梯度下降的学习率
reg_lambda = 0.01  # 正则化的强度


# 咱们先顶一个一个函数来画决策边界
def plot_decision_boundary(pred_func):
    # 设定最大最小值，附加一点点边缘填充
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # 用预测函数预测一下
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 然后画出图
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

#帮助我们在数据集上估算总体损失的函数
def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    #正向传播，计算预测值
    z1 = X.dot(W1) + b1
    a1 = np.tanh(z1)  # 隐藏层激活函数
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)  
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # 输出层激活函数softmax
    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #在损失上加上正则项（可选）
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

# 预测输出（0或1）
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # 正向传播
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)


# 这个函数为神经网络学习参数并且返回模型
# - nn_hdim: 隐藏层的节点数
# - num_passes: 通过训练集进行梯度下降的次数
# - print_loss: 如果是True, 那么每1000次迭代就打印一次损失值
def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # 用随机值初始化参数。我们需要学习这些参数
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # 这是我们最终要返回的数据
    model = {}

    # 梯度下降
    for i in range(0, num_passes):

        # 正向传播
        z1 = X.dot(W1) + b1
        a1 = np.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # 反向传播
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # 添加正则项 (b1 和 b2 没有正则项)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 梯度下降更新参数
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # 为模型分配新的参数
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # 选择性地打印损失
        # 这种做法很奢侈，因为我们用的是整个数据集，所以我们不想太频繁地这样做
        if print_loss and i % 1000 == 0:
            print
            "Loss after iteration %i: %f" % (i, calculate_loss(model))

    return model


# 生成数据集
if __name__ == "__main__":
    # 搭建一个3维隐藏层的模型
    model = build_model(3, print_loss=True)

    # 画出决策边界
    plot_decision_boundary(lambda x: predict(model, x))
    plt.title("Decision Boundary for hidden layer size 3")
    plt.show();



