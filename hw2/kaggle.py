import numpy as np
import pandas as pd
from pandas import DataFrame
# import matplotlib.pyplot as plt
import warnings

# import math
# import torch
# from torch.nn.functional import logsigmoid

# plt.switch_backend('agg')

warnings.filterwarnings('ignore')

location = 'IA2-train.csv'
location2 = 'IA2-dev.csv'
location_test = 'IA2-test-small-v2-X.csv'
data1 = pd.read_csv(location)
data2 = pd.read_csv(location2)

data_test = pd.read_csv(location_test)


def normalize_data(data):
    temp_age = data['Age']
    temp_annual_pre = data['Annual_Premium']
    temp_vintage = data['Vintage']

    data = data.drop('Age', axis=1)
    data = data.drop('Annual_Premium', axis=1)
    data = data.drop('Vintage', axis=1)

    temp_age = temp_age.astype('float')
    temp_age = (temp_age - temp_age.mean()) / temp_age.std()

    temp_annual_pre = temp_annual_pre.astype('float')
    temp_annual_pre = (temp_annual_pre -
                       temp_annual_pre.mean()) / temp_annual_pre.std()

    temp_vintage = temp_vintage.astype('float')
    temp_vintage = (temp_vintage - temp_vintage.mean()) / temp_vintage.std()

    data = pd.concat([data, temp_age], axis=1)
    data = pd.concat([data, temp_annual_pre], axis=1)
    data = pd.concat([data, temp_vintage], axis=1)

    return data


data1 = normalize_data(data1)
X = data1.drop('Response', 1)
Y = data1['Response']

X_t = normalize_data(data_test)

data2 = normalize_data(data2)
X2 = data2.drop('Response', 1)
Y2 = data2['Response']


def sigmoid(X, W):
    sig_value = 1. / (1. + np.exp(-np.matmul(X, W)))
    return sig_value


def sigmoid_min(X, W):
    sig_value = np.exp(-np.matmul(X, W)) / (1 + np.exp(-np.matmul(X, W)))
    return sig_value


def loss_function(X, Y, W, regularizationParameter):
    lambta = regularizationParameter
    N = len(Y)
    sig_y = sigmoid(X, W)
    min_sig_y = sigmoid_min(X, W)
    s = -Y * np.log(sig_y) - (1 - Y) * np.log(min_sig_y)
    tempW = np.matmul(W.T, W) - W[0] * W[0]
    lw = lambta * tempW
    loss = s.sum() / N + lw

    return loss


def L2GD(X, Y, W, learningRate, regularizationParameter, iteration):
    N = len(W)
    alpha = learningRate
    lambta = regularizationParameter
    stopping_threshold = 0.001

    loss = loss_function(X, Y, W, lambta)

    i = 0
    while iteration:
        g = Y - sigmoid(X, W)
        gradient = X.mul(g, axis=0)
        aa = (alpha / N) * gradient.sum()
        W += aa

        for j in range(1, N):
            W[j] = W[j] - alpha * lambta * W[j]

        cur_loss = loss_function(X, Y, W, lambta)

        if loss - cur_loss <= stopping_threshold and iteration < 9500:
            loss = cur_loss
            break
        loss = cur_loss
        i += 1
        # print("# of iter:  ", i)
        # print(loss)

        iteration -= 1

    return W, loss


# print('------------------\n')
# print('part 1  \n')
# print('------------------\n')


def show_part1_plt(parameter, learning_rate):
    print("training regularizationParameter :  ", parameter)
    lr = learning_rate
    ylen = len(X_t)
    par_ma = parameter
    iter = 10000
    # accurate_num = 0
    W = np.ones(X.shape[1])
    train_predition = []
    weight, loss = L2GD(X, Y, W, lr, par_ma, iter)

    y_hat = sigmoid(X_t, weight)
    min_y_hat = sigmoid_min(X_t, weight)
    ids = []
    for i in range(0, ylen):
        ids.append(i)
        if y_hat[i] > min_y_hat[i]:
            train_predition.append(1)
        elif y_hat[i] < min_y_hat[i]:
            train_predition.append(0)

    # accuracy = accurate_num / ylen

    print(ids)
    print("\n")
    print(len(X))
    print("\n")
    print(len(train_predition))
    print("\n")
    print("train pre:   \n", train_predition)
    print("\n")
    return ids, train_predition


def show_part1_small_plt(parameter, learning_rate):
    print("valdation regularizationParameter :  ", parameter)
    lr = learning_rate
    ylen = len(X_t)
    par_ma = parameter
    iter = 10000
    # accurate_num = 0
    dev_predition = []
    W = np.ones(X2.shape[1])
    weight, loss = L2GD(X2, Y2, W, lr, par_ma, iter)

    y_hat = sigmoid(X_t, weight)
    min_y_hat = sigmoid_min(X_t, weight)
    ids = []
    for i in range(0, ylen):
        ids.append(i)
        if y_hat[i] > min_y_hat[i]:
            dev_predition.append(1)
        elif y_hat[i] < min_y_hat[i]:
            dev_predition.append(0)
    # accuracy = accurate_num / ylen
    print(len(X2))
    print("\n")
    print(len(dev_predition))
    print("\n")
    print("train pre:   \n", dev_predition)
    print("\n")
    return ids, dev_predition


# print("train data:    \n")
# train_ids, train_pre = show_part1_plt(0.1, 0.01)
# t_id = np.array(train_ids)[:, np.newaxis]
# t_pre = np.array(train_pre)[:, np.newaxis]
# concat_array = np.concatenate((t_id, t_pre), axis=1)
# dt = DataFrame(concat_array, columns=["ID", "Response"])
# dt.to_csv("Chengxu_Xu_kaggle.csv", index=False)

print("dev data:    \n")
dev_ids, dev_pre = show_part1_small_plt(0.01, 0.01)
d_id = np.array(dev_ids)[:, np.newaxis]
d_pre = np.array(dev_pre)[:, np.newaxis]
concat_array = np.concatenate((d_id, d_pre), axis=1)
d_dt = DataFrame(concat_array, columns=["ID", "Response"])
d_dt.to_csv("Chengxu_Xu_kaggle1.csv", index=False)

# def generate_plot(data, lr, picname):
#     fig = plt.figure()
#     plt.plot(data[0], data[1], label='learning rate:' + str(lr))
#     # 1.a
#     plt.title('Accuracy vs Regularization Parameter,learning rate = ' +
#               str(lr))

#     plt.xlabel('Lambda', fontsize=14)
#     plt.ylabel('Accuracy', fontsize=14)
#     plt.legend()
#     picname = picname + '{}.png'.format(lr)
#     plt.savefig(picname)
#     plt.close(fig)

# lambta_list = []

# accuracy = []
# accuracy2 = []

# learning_rate = 0.01
# for i in range(-3, 4):
#     params = pow(10, i)
#     lambta_list.append(params)

#     accuracy.append(show_part1_plt(params, learning_rate))
#     accuracy2.append(show_part1_small_plt(params, learning_rate))

# generate_plot([lambta_list, accuracy], learning_rate, 'part_1_train')
# generate_plot([lambta_list, accuracy2], learning_rate, 'part_1_small')
