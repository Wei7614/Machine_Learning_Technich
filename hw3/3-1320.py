from Cart import Cart
from RandomForest import RandomForest

def load_data(file_path):
    f = open(file_path)
    x = []
    y = []
    for line in f.readlines():
        line_list = line.replace('\n', '').split(' ')
        x.append([float(line_list[value]) for value in range(len(line_list) - 1)])
        y.append(float(line_list[-1]))
    return x, y


def estimate_error(y_predict, ys):
    error_y = 0
    for idx, y in enumerate(ys):
        error_y += 1 if y_predict[idx] - y != 0 else 0
    return error_y/len(ys)


def main():
    x_train, y_train = load_data('./hw3_dectree_train.dat.txt')
    x_test, y_test = load_data('./hw3_dectree_test.dat.txt')
    cart = Cart(x_train, y_train)
    cart.build_tree()

    # q13
    print("q13: branch count =", cart.get_branch_count())

    # q14
    predict_y_train = cart.predict(x_train)
    error_y_train = estimate_error(predict_y_train, y_train)
    print('q14: e_in =', error_y_train)

    # q15
    predict_y_test = cart.predict(x_test)
    error_y_test = estimate_error(predict_y_test, y_test)
    print('q15: e_out =', error_y_test)

    # 實驗次數
    T = 10
    # 幾棵樹
    tree_limit = 300
    e_g_in = 0
    e_in = 0
    e_out = 0
    e_in_prune = 0
    e_out_prune = 0
    for t in range(T):
        rf = RandomForest(x_train, y_train)
        rf.build_trees(tree_limit)

        # 做aggregation的e_in（取得每一個小g_e_in by 每一棵樹）
        for tree in rf.trees:
            g_in = 0
            t_predict = tree.predict(x_train)
            for idx, y in enumerate(y_train):
                g_in += 1 if t_predict[idx] - y != 0 else 0
            e_g_in += g_in/len(y_train)

        predict_y_train_rf = rf.predict(x_train)
        error_y_train_rf = estimate_error(predict_y_train_rf, y_train)
        e_in += error_y_train_rf

        predict_y_test_rf = rf.predict(x_test)
        error_y_test_rf = estimate_error(predict_y_test_rf, y_test)
        e_out += error_y_test_rf

        # 實作prune = True的Random Forest
        rf_prune = RandomForest(x_train, y_train)
        rf_prune.build_trees(tree_limit, True)
        predict_y_tran_rf_prune = rf_prune.predict(x_train)
        error_y_train_rf_prune = estimate_error(predict_y_tran_rf_prune, y_train)
        e_in_prune += error_y_train_rf_prune

        predict_y_test_rf_prune = rf_prune.predict(x_test)
        error_y_test_rf_prune = estimate_error(predict_y_test_rf_prune, y_test)
        e_out_prune += error_y_test_rf_prune

    print("q16: e_g_in = ", e_g_in / (T * tree_limit))
    print('q17: e_in =', e_in / T)
    print('q18: e_out =', e_out / T)
    print('q19: e_in =', e_in_prune/T)
    print('q20: e_out =', e_out_prune/T)


if __name__ == '__main__':
    main()
