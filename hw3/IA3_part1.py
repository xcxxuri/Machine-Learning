import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
plt.switch_backend('agg')

# loading csv file
location = 'IA2-train.csv'
location2 = 'IA2-dev.csv'
data1 = pd.read_csv(location)
data2 = pd.read_csv(location2)

print('------------------\n')
print('part 1  \n')
print('------------------\n')


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


# data pre-processing
data1 = normalize_data(data1)
X = data1.drop('Response', 1)
Y = data1['Response']
for i in range(len(Y)):
    if Y.iloc[i] == 0:
        Y.iloc[i] = -1

data2 = normalize_data(data2)
X2 = data2.drop('Response', 1)
Y2 = data2['Response']
for i in range(len(Y2)):
    if Y2.iloc[i] == 0:
        Y2.iloc[i] = -1


# get accuracy function
def get_accuracy(X, Y, W, W_ave):
    M = X.shape[0]
    N = len(Y)
    W_acc = 0
    W_ave_acc = 0
    for i in range(M):
        Xi = X.iloc[i]
        W_Y_hat = np.sum(np.matmul(Xi, W))
        W_ave_Y_hat = np.sum(np.matmul(Xi, W_ave))
        # counting correct predictions
        if Y[i] * W_Y_hat > 0:
            W_acc += 1
        if Y[i] * W_ave_Y_hat > 0:
            W_ave_acc += 1

    return W_acc / N, W_ave_acc / N


# Algorithm 1
def AP(X, Y, W, W_ave, max_iteration):
    N = len(W)
    M = X.shape[0]
    iteration = 0

    # W， W_ave and example counter s initialize
    W_acc_array = []
    W_ave_acc_array = []
    s = 1
    # part1.b parameter initialize
    W_ave_mi = 0
    W_ave_ma = 0
    W_mi = 0
    W_ma = 0

    while iteration < max_iteration:
        for i in range(M):
            Xi = X.iloc[i]
            if Y[i] * np.sum(np.matmul(Xi, W)) <= 0:
                for j in range(N):
                    W[j] = W[j] + Y[i] * Xi[j]
            W_ave = (s * W_ave + W) / (s + 1)
            s += 1

        W_acc, W_ave_acc = get_accuracy(X, Y, W, W_ave)
        W_acc_array.append(W_acc)
        W_ave_acc_array.append(W_ave_acc)

        if W_acc > W_ma:
            W_ma = W_acc
            W_mi = iteration

        if W_ave_acc > W_ave_ma:
            W_ave_ma = W_ave_acc
            W_ave_mi = iteration
        # print("accuracy: ", W_acc, W_ave_acc)
        iteration += 1

    return W_acc_array, W_ave_acc_array, W_ave_mi, W_ave_ma, W_mi, W_ma


def generate_W_accuracy_plot(data, iter, picname):
    fig = plt.figure()
    plt.plot(iter, data, label="onine perceptron")
    plt.title("Accuracy with perceptron W", fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


def generate_W_ave_accuracy_plot(data, iter, picname):
    fig = plt.figure()
    plt.plot(iter, data, label="average perceptron")
    plt.title("Accuracy with average perceptron W", fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


# --------------------------------
index_array = []
max_iteration = 100
for i in range(max_iteration):
    index_array.append(i + 1)


def show_train(X, Y, max_iter=max_iteration, index=index_array):
    W = np.ones(X.shape[1])
    W_ave = np.ones(X.shape[1])
    for i in range(len(W)):
        W[i] = 0
        W_ave[i] = 0
    w_acc_array, W_ave_acc_array, W_ave_mi, W_ave_ma, W_mi, W_ma = AP(
        X, Y, W, W_ave, max_iter)
    generate_W_accuracy_plot(w_acc_array, index, "part1_train_w")
    generate_W_ave_accuracy_plot(W_ave_acc_array, index, "part1_train_w_ave")


def show_dev(X, Y, max_iter=max_iteration, index=index_array):
    W = np.ones(X.shape[1])
    W_ave = np.ones(X.shape[1])
    for i in range(len(W)):
        W[i] = 0
        W_ave[i] = 0
    w_acc_array, W_ave_acc_array, W_ave_mi, W_ave_ma, W_mi, W_ma = AP(
        X, Y, W, W_ave, max_iter)

    print("part 1 b : \n")
    print("max online perceptron accuracy:             ", W_ma)
    print("max online perceptron accuracy iteration:   ", W_mi)
    print("max average perceptron accuracy:            ", W_ave_ma)
    print("max average perceptron accuracy iteration:  ", W_ave_mi)

    generate_W_accuracy_plot(w_acc_array, index, "part1_dev_w")
    generate_W_ave_accuracy_plot(W_ave_acc_array, index, "part1_dev_w_ave")


show_train(X, Y)

show_dev(X2, Y2)
