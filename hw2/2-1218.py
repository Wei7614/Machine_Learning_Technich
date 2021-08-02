import numpy as np
import math

train_data_source = np.loadtxt('./hw2_adaboost_train.dat.txt')
test_data_source = np.loadtxt('./hw2_adaboost_test.dat.txt')


def get_x_y(data_source):
    return data_source[:, 0:data_source.shape[1] - 1], data_source[:, data_source.shape[1] - 1:data_source.shape[1]]


x_train, y_train = get_x_y(train_data_source)
x_test, y_test = get_x_y(test_data_source)


def sign(s):
    return -1 if s < 0 else 1


def decision_stump(s, x, theta):
    return s * sign(x - theta)


def get_thresholds(x_series):
    x_series = sorted(x_series)
    thresholds = [math.inf * -1]
    for idx, x in enumerate(x_series):
        if (idx + 1) < len(x_series):
            thresholds.append((x + x_series[idx + 1]) / 2)
    return thresholds


def re_weight(weight):
    pass


# training挑出e最小的s, threshold, 跟feature
def training(x_data, y, weight):
    min_e_in = math.inf
    best_s = 0
    best_col = 0
    best_threshold = math.inf * -1
    for col in range(x_data.shape[1]):
        features = x_data[:, col:col + 1].tolist()
        features = [value[0] for value in features]
        thresholds = get_thresholds(features)
        s_s = [-1, 1]
        for threshold in thresholds:
            for s in s_s:
                e_sum = 0
                for row in range(x_data.shape[0]):
                    x = x_data[row, col:col + 1]
                    g = decision_stump(s, x, threshold)
                    if y[row, 0] != g:
                        # 錯誤的分數要weighted
                        e_sum += weight[row, 0]
                if min_e_in > e_sum:
                    min_e_in = e_sum
                    best_threshold = threshold
                    best_s = s
                    best_col = col
    return min_e_in/np.sum(weight), best_s, best_col, best_threshold


def optimal_reweighting(x_data, y_data, weight, threshold, s, column, diamond):
    for row in range(x_data.shape[0]):
        x = x_data[row, column]
        g = decision_stump(s, x, threshold)
        if y_data[row, 0] != g:
            # 錯的就放大
            weight[row, 0] *= diamond
        else:
            # 對的就縮小
            weight[row, 0] /= diamond


def error_estimate(g, x_data, y_data):
    error = 0
    s, col, threshold = g
    for row in range(x_data.shape[0]):
        value = decision_stump(s, x_data[row, col], threshold)
        error += 1 if value != y_data[row, 0] else 0
    return error/x_data.shape[0]


def boost_error_estimate(alphas, x_data, y_data, g_s):
    boost_error = 0
    for row in range(x_data.shape[0]):
        G = 0
        for idx, alpha in enumerate(alphas):
            s, col, threshold = g_s[idx]
            G += alpha * decision_stump(s, x_data[row, col], threshold)
        boost_error += 1 if sign(G) != y_data[row, 0] else 0
    return boost_error/x_data.shape[0]


def main():
    iterator_times = 300
    g_s = []
    alphas = []
    # 課程裡面講的u
    weight = np.ones((x_train.shape[0], 1)) * (1 / x_train.shape[0])
    weight_sum = []
    e_in_min = math.inf
    for i in range(iterator_times):
        e_in, best_s, best_col, best_threshold = training(x_train, y_train, weight)
        e_in_min = min(e_in_min, e_in)
        weight_sum.append(np.sum(weight))
        g_s.append((best_s, best_col, best_threshold))
        # 為了要得到發散的 g 所以要有scaling_factor跟optimal_reweighting
        scaling_factor = math.sqrt((1 - e_in) / e_in)
        optimal_reweighting(x_train, y_train, weight, best_threshold, best_s, best_col, scaling_factor)
        alphas.append(math.log(scaling_factor))

    # q12
    e_in_g_1 = error_estimate(g_s[0], x_train, y_train)
    print("q12 : g1 e_in = {0:.2f}".format(e_in_g_1))

    # q13 q18
    error_in_G = boost_error_estimate(alphas, x_train, y_train, g_s)
    error_out_G = boost_error_estimate(alphas, x_test, y_test, g_s)

    print("q13 : e_in_G = {0:.2f}".format(error_in_G))

    # q14
    print('q14 : u_2 = {0:.2f}'.format(weight_sum[1]))

    # q15
    print("q15 : u_T = {0:.2f}".format(weight_sum[299]))

    # q16
    print("q16 : min epsilon = {0:.2f}".format(e_in_min))

    # q17
    e_out_g_1 = error_estimate(g_s[0], x_test, y_test)
    print("q17 : q1 e_out = {0:.2f}".format(e_out_g_1))

    # q18
    print("q18 : e_out_G = {0:.2f}".format(error_out_G))


if __name__ == '__main__':
    main()
