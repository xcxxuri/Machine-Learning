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


data1 = normalize_data(data1)
X = data1.drop('Response', 1)
Y = data1['Response']

data2 = normalize_data(data2)
X2 = data2.drop('Response', 1)
Y2 = data2['Response']


def sigmoid(X, W):
    sig_value = 1. / (1. + np.exp(-np.matmul(X, W)))
    return sig_value


def sigmoid_min(X, W):
    sig_value = np.exp(-np.matmul(X, W)) / (1 + np.exp(-np.matmul(X, W)))
    return sig_value


def loss_function_2(X, Y, W, regularizationParameter):
    lambta = regularizationParameter
    N = len(Y)
    sig_y = sigmoid(X, W)
    min_sig_y = sigmoid_min(X, W)

    s = -Y * np.log(sig_y) - (1 - Y) * np.log(min_sig_y)
    tempW = np.absolute(W)
    sumTempW = tempW.sum() - W[0]

    lw = lambta * sumTempW
    loss = s.sum() / N + lw

    return loss


def L1GD(X, Y, W, learningRate, regularizationParameter, iteration):
    N = len(W)
    alpha = learningRate
    lambta = regularizationParameter
    stopping_threshold = 0.001
    i = 0

    loss = loss_function_2(X, Y, W, lambta)

    while iteration:
        g = Y - sigmoid(X, W)
        gradient = X.mul(g, axis=0)
        aa = (alpha / N) * gradient.sum()
        W += aa

        for j in range(1, N):
            W[j] = np.sign(W[j]) * max(np.abs(W[j]) - alpha * lambta, 0)

        cur_loss = loss_function_2(X, Y, W, lambta)

        if loss - cur_loss <= stopping_threshold and iteration < 2950:
            loss = cur_loss
            break

        if cur_loss > 3000 and iteration < 2950:
            loss = cur_loss
            break

        loss = cur_loss
        i += 1
        # print("# of iter:  ", i)
        # print(loss)

        iteration -= 1

    return W, loss


def show_part2_plt(parameter, learning_rate):
    print("training regularizationParameter :  ", parameter)
    lr = learning_rate
    ylen = len(Y)
    par_ma = parameter
    iter = 3000
    accurate_num = 0
    W = np.ones(X.shape[1])
    weight, loss = L1GD(X, Y, W, lr, par_ma, iter)

    # 2.b
    if par_ma == 0.01:
        print("---------part2.b train data----------")
        w_lambta_min = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_min[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    elif par_ma == 0.1:
        print("---------part2.b train data----------")
        w_lambta_star = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_star[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    elif par_ma == 1:
        print("---------part2.b train data----------")
        w_lambta_plus = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_plus[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    # 2.c
    sparsity = 197 - np.count_nonzero(np.around(weight, decimals=0))
    print("---------part2.c ----------")
    print("parameter =   ", par_ma)
    print("training sparsity=   ", sparsity)

    y_hat = sigmoid(X, weight)
    min_y_hat = sigmoid_min(X, weight)
    for i in range(0, ylen):
        if y_hat[i] > min_y_hat[i] and Y[i] == 1:
            accurate_num += 1
        elif y_hat[i] < min_y_hat[i] and Y[i] == 0:
            accurate_num += 1
    accuracy = accurate_num / ylen

    print("---------part2.a ----------")
    print("accuracy=   ", accuracy, "\n")
    return accuracy, sparsity


def show_part2_small_plt(parameter, learning_rate):
    print("valdation regularizationParameter :  ", parameter)
    lr = learning_rate
    ylen = len(Y2)
    par_ma = parameter
    iter = 3000
    accurate_num = 0
    W = np.ones(X2.shape[1])
    weight, loss = L1GD(X2, Y2, W, lr, par_ma, iter)

    # 2.b
    if par_ma == 0.01:
        print("---------part2.b dev data----------")
        w_lambta_min = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_min[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    elif par_ma == 0.1:
        print("---------part2.b dev data----------")
        w_lambta_star = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_star[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    elif par_ma == 1:
        print("---------part2.b dev data----------")
        w_lambta_plus = L1GD(X2, Y2, W, lr, par_ma, iter)
        new_w = list(w_lambta_plus[0])
        new_w = list(map(abs, new_w))
        top_list = np.argsort(new_w)[-5:]
        columns = data1.columns

        for i in range(5):
            index = top_list[i]
            column = columns[index]
            print('{}: {}'.format(column, new_w[index]))
        print('\n')

    # 2.c
    sparsity = 197 - np.count_nonzero(np.around(weight, decimals=0))
    print("---------part2.c ----------")
    print("parameter =   ", par_ma)
    print("validation sparsity=   ", sparsity)

    y_hat = sigmoid(X2, weight)
    min_y_hat = sigmoid_min(X2, weight)
    for i in range(0, ylen):
        if y_hat[i] > min_y_hat[i] and Y2[i] == 1:
            accurate_num += 1
        elif y_hat[i] < min_y_hat[i] and Y2[i] == 0:
            accurate_num += 1
    accuracy = accurate_num / ylen

    print("---------part2.a ----------")
    print("accuracy=   ", accuracy, "\n")
    return accuracy, sparsity


def generate_accuracy_plot(data, lr, picname):
    fig = plt.figure()
    plt.plot(data[0], data[1], label='learning rate:' + str(lr))

    plt.title('Accuracy vs Regularization Parameter, learning rate,  = ' +
              str(lr))

    plt.xlabel('Lambda', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


def generate_sparsity_plot(data, lr, picname):
    fig = plt.figure()
    plt.plot(data[0], data[1], label='learning rate:' + str(lr))

    plt.title('Sparsity vs Regularization Parameter, learning rate = ' +
              str(lr))
    plt.xlabel('Lambda', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend()
    plt.savefig(picname)
    plt.close(fig)


lambta_list_p2 = []
# 2.a
accuracy_p2 = []
accuracy2_p2 = []
# 2.c
sparsity_p2 = []
sparsity2_p2 = []

learning_rate = 0.01
for i in range(-3, 4):
    params = pow(10, i)
    lambta_list_p2.append(params)

    train_accuracy, train_sparsity = show_part2_plt(params, learning_rate)
    small_accuracy, small_sparsity = show_part2_small_plt(
        params, learning_rate)

    # 2.a
    accuracy_p2.append(train_accuracy)
    accuracy2_p2.append(small_accuracy)

    # 2.c
    sparsity_p2.append(train_sparsity)
    sparsity2_p2.append(small_sparsity)

# 2.a
generate_accuracy_plot([lambta_list_p2, accuracy_p2], learning_rate,
                       'part_2_train')
generate_accuracy_plot([lambta_list_p2, accuracy2_p2], learning_rate,
                       'part_2_dev')

# 2.c
generate_sparsity_plot([lambta_list_p2, sparsity_p2], learning_rate,
                       'part_2_train_sparsity')
generate_sparsity_plot([lambta_list_p2, sparsity2_p2], learning_rate,
                       'part_2_dev_sparsity')
