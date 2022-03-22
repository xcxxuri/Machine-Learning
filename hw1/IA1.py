import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

# plt.switch_backend('agg')

warnings.filterwarnings('ignore')
# loading csv file
location = 'IA1_train.csv'
location2 = 'IA1_dev.csv'
data1 = pd.read_csv(location)
data_train = pd.read_csv(location2)


def data_init(d):
    data = d
    data = data.drop(columns=['id'])

    # part0
    # split the date feature into month, day, year
    monthDayYear = data['date'].str.split('/',expand=True) # Convert series type to DataFrame type
    monthDayYear.columns = ['month', 'day', 'year']
    data = pd.concat([data, monthDayYear],axis = 1) # connect monthDayYear's column into data
    data = data.drop(columns=['date']) # delete date column

    # add a dummy feature
    data.insert(data.shape[1],'dummy',1)

    # construct age_since_renovated
    for x in range(len(data)):
        data.loc[[x],'year'] = int(data.loc[x,'year']) + 2000
        data.loc[[x],'yr_renovated'] = int(data.loc[x,'year']) - data.loc[x,'yr_built'] if data.loc[x, 'yr_renovated'] == 0 else int(data.loc[x,'year']) - data.loc[x,'yr_renovated']

    data = data.rename(columns={'yr_renovated' : 'age_since_renovated'})

    waterfront = data['waterfront']
    dummy = data['dummy']
    price = data['price']
    data = data.drop(columns='waterfront')
    data = data.drop(columns='dummy')
    data = data.drop(columns='price')
    data = data.astype('float')
    data = (data - data.mean()) / data.std()

    data = pd.concat([data, waterfront], axis = 1)
    data = pd.concat([data, dummy], axis = 1)
    data = pd.concat([data, price],axis = 1)
    # print(data)
    return data


def non_normal_data_init(d):
    data = d
    data = data.drop(columns=['id'])

    # split the date feature into month, day, year
    monthDayYear = data['date'].str.split('/',expand=True) # Convert series type to DataFrame type
    monthDayYear.columns = ['month', 'day', 'year']
    data = pd.concat([data, monthDayYear],axis = 1) # connect monthDayYear's column into data
    data = data.drop(columns=['date']) # delete date column

    # add a dummy feature
    data.insert(data.shape[1],'dummy',1)

    # construct age_since_renovated
    for x in range(len(data)):
        data.loc[[x],'year'] = int(data.loc[x,'year']) + 2000
        data.loc[[x],'yr_renovated'] = int(data.loc[x,'year']) - data.loc[x,'yr_built'] if data.loc[x, 'yr_renovated'] == 0 else int(data.loc[x,'year']) - data.loc[x,'yr_renovated']

    data = data.rename(columns={'yr_renovated' : 'age_since_renovated'})
    data = data.astype('float')
    # print(data)
    return data
    


# part1
# a
input_data = data_init(data1)

data_input = input_data.drop('price',1)
training_data = data_input.to_numpy()
data_label = input_data['price']
lables = data_label.to_numpy()

iter = 5000 # max iteration

# BGD
def gradient_descent(input_data, target, learningrate, iteration = iter):
    m = len(target)
    loss = np.empty(iteration)
    w_num = input_data.shape[1]
    w = np.random.randn(w_num, 1)
    # input_data = input_data.astype('float')
    # w = w.astype('float')
    losses = []
    # weight_h = []
    loss_change = 0
    loss_increase = 0

    # loop for gradient descent
    for i in range (0, iteration):
        loss = 0

        output = np.dot(input_data, w)
        output = np.squeeze(output)
        se = output - target # squre error
        print(len(target))
        print(se)
        print(len(se))
        input()

        gradient = (2. / m) * np.sum(input_data.T.dot(se)) 

        loss += (1. / m) * np.sum(np.square(se))
        w = w - learningrate * gradient

        losses.append(loss)

        if np.abs((losses[i-1]- losses[i] / losses[i - 1]) < 0.01):
            loss_change += 1
        
        if losses[i] - losses[i -1] > 0:
            loss_increase += 1
        # condition to stop loop
        if losses == 0:
            break

        if i > 1000 and loss_change > 5:
            break

        if i > 1000 and loss_increase > 100:
            break
    
    # print('Training is finished!', 'lr = ', learningrate )
    return losses, w

def show_plt(learning_rate):
    lr = learning_rate
    loss, weight = gradient_descent(training_data, lables, lr)
    plt.figure()
    plt.plot(loss)
    plt.title('Loss Learning Rate =  %g' %lr)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
print('------------------\n')
print('part 1 (a) \n' )
print('------------------\n')
for i in range (-6, 2):
    show_plt(pow(10,i))


# # b
# input_train_data = data_init(data_train)
# train_data_input = input_train_data.drop('price',1)
# training_train_data = train_data_input.to_numpy()
# train_data_label = input_train_data['price']
# train_lables = train_data_label.to_numpy()

# def show_mse(learning_rate):
#     lr = learning_rate
#     loss, weight = gradient_descent(training_data, lables, lr)
#     print(' When Learning rate =  %g \n' %lr ,'MSE =',loss[len(loss)-1])

# print('------------------\n')
# print('part 1 (b) \n' )
# print('------------------\n')
# for i in range (-6, -1):
#     show_mse(pow(10,i))


# # c
# # calculating the learned weights when learning rate was 0.00001
# best_learned_loss, best_learned_weight = gradient_descent(training_data, lables, 0.00001)
# print('------------------\n')
# print('part 1 (c) \n' )
# print('------------------\n')
# for i in range (len(best_learned_weight)):
#     new_weight = np.squeeze(best_learned_weight)
#     feature = train_data_input.columns[i]
#     best_lw = new_weight[i]

#     print(feature, ':', best_lw)

# # part 2
# # a
# location3 = 'IA1_train_nonnormalized.csv'
# location4 = 'IA1_dev_nonnormalized.csv'
# data1_non_normal = pd.read_csv(location3)
# data_train_non_normal = pd.read_csv(location4)


# input_non_normal_data = non_normal_data_init(data1_non_normal)

# non_normal_data_input = input_non_normal_data.drop('price',1)
# training_non_normal_data = non_normal_data_input.to_numpy()
# non_normal_data_label = input_non_normal_data['price']
# non_lables = non_normal_data_label.to_numpy()


# def show_non_plt(learning_rate):
#     lr = learning_rate
#     loss, weight = gradient_descent(training_non_normal_data, non_lables, lr)
#     plt.figure()
#     plt.plot(loss)
#     plt.title('Non normalized train loss Learning Rate =  %g' %lr)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()
    
# print('------------------\n')
# print('part 2 (a) \n' )
# print('------------------\n')
# for i in range (-15, 2):
#     show_non_plt(pow(10,i))


# # b
# input_train_non_normal_data = non_normal_data_init(data_train_non_normal)

# non_normal_train_data_input = input_train_non_normal_data.drop('price',1)
# training_train_non_normal_data = non_normal_train_data_input.to_numpy()
# train_non_normal_data_label = input_train_non_normal_data['price']
# non_train_lables = train_non_normal_data_label.to_numpy()

# def show_non_mse(learning_rate):
#     lr = learning_rate
#     loss, weight = gradient_descent(training_train_non_normal_data, non_train_lables, lr)
#     print(' When Learning rate =  %g \n' %lr ,'MSE =',loss[len(loss)-1])
#     plt.figure()
#     plt.plot(loss)
#     plt.title('Non normalized train loss Learning Rate =  %g' %lr)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()

# print('------------------\n')
# print('part 2 (b) \n' )
# print('------------------\n')
# for i in range (-15, -9):
#     show_non_mse(pow(10,i))


# # part 3
# print('------------------\n')
# print('part 3 \n' )
# print('------------------\n')
# data_input_normal = data_init(data1)

# p3_data_input_normal = data_input_normal.drop('price', 1)
# p3_data_input_normal = p3_data_input_normal.drop('sqft_living15',1)
# p3_data = p3_data_input_normal.to_numpy()
# p3_data_lable = data_input_normal['price']
# p3_lable = p3_data_lable.to_numpy()

# # calculating the learned weights when learning rate was 0.00001
# p3_lr = 0.00001 # learning rate
# def p3_MSE(learning_rate):
#     lr = learning_rate 
#     loss, lw = gradient_descent(p3_data, p3_lable, lr)
#     print('Remove sqft_living15 feature, learning rate =  %g \n' %lr ,'MSE =',loss[len(loss)-1])
#     plt.figure()
#     plt.plot(loss)
#     plt.title('Normalized train loss with Learning Rate =  %g' %lr)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.show()
#     for i in range (len(lw)):
#         lw_p3 = np.squeeze(lw)
#         feature = p3_data_input_normal.columns[i]
#         p3_weight = lw_p3[i]
#         print(feature, ':', p3_weight)
    
# p3_MSE(p3_lr)
