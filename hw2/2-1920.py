# kernel ridge regression 就是在講 optimal w 被拆成 sum_n(beta_n * z_n)
# 之後帶入ridge regression (lambda/N) * w*w + 1/N * sum_n(y_n - w * z_n)^2
# 就可以變成 (lambda/N) * sum_n * sum_m * beta_n( beta_m( k(x_n, x_m))) +  1/N * sum_n(y_n - sum_m(beta_m * k(x_m, x_n)))^2
# beta = (lambda * I - k)^-1 * y


import numpy as np
import math

data_source = np.loadtxt('hw2_lssvm_all.dat.txt')
amount_training = 400
training_data = data_source[0:amount_training, :]
testing_data = data_source[amount_training:, :]


def get_x_y_datas(data):
    return data[:, 0: data.shape[1] - 1], data[:, data.shape[1] - 1: data.shape[1]]


x_train, y_train = get_x_y_datas(training_data)
x_test, y_test = get_x_y_datas(testing_data)


def sign(s):
    return -1 if s <= 0 else 1


def kernel_function(x_n, x_m, gamma):
    result = np.zeros((x_n.shape[0], x_m.shape[0]))
    for row_n in range(x_n.shape[0]):
        x_pri = x_n[row_n, 0:x_n.shape[1]]
        for row_m in range(x_m.shape[0]):
            x = x_m[row_m, 0:x_m.shape[1]]
            result[row_n, row_m] = np.exp(-gamma * np.sqrt(np.sum((x_pri - x) * (x_pri - x))))
    return result


def get_beta(k, l, y):
    lambda_matrix = np.identity(k.shape[0]) * l
    inverse = np.linalg.inv(lambda_matrix + k)
    result = np.dot(inverse, y)
    return result


def error_estimate(beta, k, y):
    predict = np.dot(k.T, beta)
    for idx in range(len(predict)):
        predict[idx, 0] = sign(predict[idx, 0])
    error = np.sum(np.abs(predict - y)/2.)/y.shape[0]
    return error


def main():
    gammas = [32, 2, 0.125]
    lambdas = [0.001, 1, 1000]
    min_e_in = math.inf
    for gamma in gammas:
        k = kernel_function(x_train, x_train, gamma)
        for lamda in lambdas:
            beta = get_beta(k, lamda, y_train)
            error_in = error_estimate(beta, k, y_train)
            if error_in <= min_e_in:
                min_e_in = error_in
                min_gamma = gamma
                # min_lambda = lamda
                min_beta = beta

    k_test = kernel_function(x_train, x_test, gamma)
    error_out = error_estimate(beta, k_test, y_test)
    print("min e in = {0:.2f}".format(min_e_in))
    print("min e out = {0:.2f}".format(error_out))


if __name__ == '__main__':
    main()