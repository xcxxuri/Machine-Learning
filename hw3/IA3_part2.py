import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import operator
from functools import reduce
import time

warnings.filterwarnings('ignore')
plt.switch_backend('agg')

# loading csv file
location = 'IA2-train.csv'
location2 = 'IA2-dev.csv'
data1 = pd.read_csv(location)
data2 = pd.read_csv(location2)

print('------------------\n')
print('part 2  \n')
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


def kernel_function(X1, X2, p):
    K = np.power((np.matmul(X1, X2.T)), p)
    return K


def prediction(alpha, K, Y):
    pre = np.sign(np.matmul(K, np.multiply(alpha, Y)))
    return pre


def get_accuracy(Y_hat, Y):
    N = len(Y)
    acc = 0
    for i in range(N):
        if Y_hat[i] == Y[i]:
            acc += 1
    # print("accuracy:  ", acc / N)
    return acc / N


def kernel_perceptron(X, Y, X2, Y2, p, max_iteration):
    N = len(X)
    iteration = 0
    alpha = np.zeros(N)
    acc = []
    K = kernel_function(X, X, p)
    best_accuracy = 0

    while iteration < max_iteration:
        for i in range(N):
            u = np.sign(np.dot(K[i], np.multiply(alpha, Y)))
            if Y[i] * u <= 0:
                alpha[i] += 1

        # print(alpha)
        # input()

        pred = prediction(alpha, kernel_function(X2, X, p), Y)
        if Y2 is not None:
            acc.append(get_accuracy(pred, Y2))

        if get_accuracy(pred, Y2) > best_accuracy:
            best_accuracy = get_accuracy(pred, Y2)
        iteration += 1
    return alpha, acc, pred, best_accuracy


def generate_accuracy_plot(tdata, vdata, iter, picname, p):
    fig = plt.figure()
    plt.plot(iter, tdata, label="training accuracy")
    plt.plot(iter, vdata, label="validation accuracy")
    plt.title("Accuracy with p = " + str(p + 1), fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


print("------------part2a.ab-------------")
index_array = []
max_iteration = 100
for i in range(max_iteration):
    index_array.append(i + 1)


def show(X, Y, X2, Y2, max_iter=max_iteration, index=index_array):

    for i in range(0, 5):
        t_acc_array = []
        v_acc_array = []
        t_alpha, t_acc, t_pre, best_t_accuracy = kernel_perceptron(
            X, Y, X, Y, i + 1, max_iteration)
        t_acc_array.append(t_acc)
        v_alpha, v_acc, v_pre, best_v_accuracy = kernel_perceptron(
            X, Y, X2, Y2, i + 1, max_iteration)
        v_acc_array.append(v_acc)

        # part2a.b
        print("Best training accuracy for p =  " + str(i + 1) + " :   ",
              best_t_accuracy)
        print("Best valdation accuracy for p =  " + str(i + 1) + " :   ",
              best_v_accuracy)

        generate_accuracy_plot(reduce(operator.add, t_acc_array),
                               reduce(operator.add, v_acc_array), index,
                               "part2_a p = " + str(i + 1), i)


# part2a.ab
show(X, Y, X2, Y2)

# part2a.c
print("------------part2a.c-------------")

c_index_array = []
c_max_iteration = 100
for i in range(c_max_iteration):
    index_array.append(i + 1)
X10 = X[0:10:]

Y10 = Y[0:10:]
X100 = X[0:100:]
Y100 = Y[0:100:]
X1000 = X[0:1000:]
Y1000 = Y[0:1000:]
X_3 = [X10, X100, X1000, X2]
Y_3 = [Y10, Y100, Y1000, Y2]


def generate_time_plot(time, size, picname):
    fig = plt.figure()
    plt.plot(size, time, label="p=1")

    plt.title("Running time with p = 1", fontsize=16)
    plt.xlabel('Data size', fontsize=16)
    plt.ylabel('Runtime', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


def ac_show(X, Y, max_iter=c_max_iteration, index=c_index_array):
    rtime = []
    dsize = []
    for i in range(0, 4):
        start = time.time()
        Xn = X[i]
        Yn = Y[i]
        N = len(Yn)
        t_alpha, t_acc, t_pre, best_t_accuracy = kernel_perceptron(
            Xn, Yn, Xn, Yn, 1, c_max_iteration)
        end = time.time()
        rtime.append(end - start)
        dsize.append(pow(10, i + 1))
        print("datasize =   ", pow(10, i + 1))
        print("run time =   ", end - start)

    generate_time_plot(rtime, dsize, "part2_c")


ac_show(X_3, Y_3)

print("------------part2b.a-------------")
ba_index_array = []
ba_max_iteration = 100
for i in range(ba_max_iteration):
    ba_index_array.append(i + 1)


def batch_kernel_perceptron(X, Y, X2, Y2, p, ba_max_iteration):
    N = len(X)
    iteration = 0
    alpha = np.zeros(N)
    acc = []
    K = kernel_function(X, X, p)

    while iteration < ba_max_iteration:
        batch = np.zeros(len(Y))
        for i in range(N):
            u = np.sign(np.dot(K[i], np.multiply(alpha, Y)))
            if Y[i] * u <= 0:
                batch[i] += 1

        alpha += batch

        pred = prediction(alpha, kernel_function(X2, X, p), Y)
        if Y2 is not None:
            acc.append(get_accuracy(pred, Y2))

        iteration += 1
    return alpha, acc,


def b_generate_accuracy_plot(tdata, vdata, iter, picname, p):
    fig = plt.figure()
    plt.plot(iter, tdata, label="training accuracy")
    plt.plot(iter, vdata, label="validation accuracy")
    plt.title("Accuracy with p = " + str(p), fontsize=16)
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


def ba_show(X, Y, X2, Y2, max_iter=ba_max_iteration, index=ba_index_array):

    t_acc_array = []
    v_acc_array = []
    t_alpha, t_acc = batch_kernel_perceptron(X, Y, X, Y, 1, ba_max_iteration)
    t_acc_array.append(t_acc)
    v_alpha, v_acc = batch_kernel_perceptron(X, Y, X2, Y2, 1, ba_max_iteration)
    v_acc_array.append(v_acc)

    b_generate_accuracy_plot(reduce(operator.add, t_acc_array),
                             reduce(operator.add, v_acc_array), ba_index_array,
                             "part2_b p = 1 ", 1)


# part2b.a
ba_show(X, Y, X2, Y2)

# part2b.b
print("------------part2b.b-------------")

bc_index_array = []
bc_max_iteration = 100
for i in range(bc_max_iteration):
    index_array.append(i + 1)
X10 = X[0:10:]

Y10 = Y[0:10:]
X100 = X[0:100:]
Y100 = Y[0:100:]
X1000 = X[0:1000:]
Y1000 = Y[0:1000:]
X_3 = [X10, X100, X1000, X2]
Y_3 = [Y10, Y100, Y1000, Y2]


def generate_time_plot(time, size, picname):
    fig = plt.figure()
    plt.plot(size, time, label="p=1")

    plt.title("Running time with p = 1", fontsize=16)
    plt.xlabel('Data size', fontsize=16)
    plt.ylabel('Runtime', fontsize=16)

    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


def bc_show(X, Y, max_iter=bc_max_iteration, index=bc_index_array):
    rtime = []
    dsize = []
    for i in range(0, 4):
        start = time.time()
        Xn = X[i]
        Yn = Y[i]
        N = len(Yn)
        t_alpha, t_acc = batch_kernel_perceptron(Xn, Yn, Xn, Yn, 1,
                                                 bc_max_iteration)
        end = time.time()
        rtime.append(end - start)
        dsize.append(pow(10, i + 1))
        print("datasize =   ", pow(10, i + 1))
        print("run time =   ", end - start)

    generate_time_plot(rtime, dsize, "part2_bc")


bc_show(X_3, Y_3)
