import math


class Cart:
    is_prune = False

    # private variable
    _x_data = None
    _y_data = None
    _branches = None
    _branch_count = 0

    def __init__(self, x, y):
        self._x_data = x
        self._y_data = y

    def _sign(self, s):
        return -1 if s < 0 else 1

    def _decision_stump(self, s, theta, x):
        return s * self._sign(x - theta)

    def _get_thetas(self, x_s):
        thetas = []
        if len(x_s) != 0:
            feature_count = len(x_s[0])
            for feature_idx in range(feature_count):
                thetas_inner = []
                sorted_xs = sorted(x_s, key=lambda x: x[feature_idx])
                for n in range(len(sorted_xs)):
                    x_middle = (sorted_xs[n][feature_idx] + sorted_xs[n + 1][feature_idx]) / 2. if n + 1 < len(
                        sorted_xs) else sorted_xs[n][feature_idx]
                    thetas_inner.append(x_middle)
                thetas.append(thetas_inner)
        return thetas

    # gini index = 1-sum_k((sum_n(y_n == k)/N)^2)
    # k : 種類, n：資料index, N:總資料數
    def _compute_gini_index(self, ys):
        pos = 0
        neg = 0
        for idx, g in enumerate(ys):
            if g == 1:
                pos += 1
            else:
                neg += 1
        if len(ys) == 0:
            return 0
        gini_index = 1 - (float(pos / len(ys))) ** 2 - (float(neg / len(ys))) ** 2
        return gini_index

    def _create_branch(self, x, y):
        branches = []
        if not self.is_prune:
            # 還沒fully grown就要切
            if len(y) != abs(sum(y)):
                branch, left, right = self._branching(x, y)
                branches.append(branch)

                branch_left = self._create_branch(left[0], left[1])
                branches.append(branch_left)

                branch_right = self._create_branch(right[0], right[1])
                branches.append(branch_right)
            else:
                branches.append(y[0])
        else:
            # 有prune的時候就只有一層
            branch, left, right = self._branching(x, y)
            branches.append(branch)
            if sum(left[1]) >= 0:
                branches.append([1])
            else:
                branches.append([-1])

            if sum(right[1]) >= 0:
                branches.append([1])
            else:
                branches.append([-1])
        return branches

    def _fit_data(self, x, branches):
        if len(branches) >= 3:
            # s, theta, feature_idx
            branch = branches[0]
            s = branch[0]
            theta = branch[1]
            feature_idx = branch[2]
            y = self._decision_stump(s, theta, x[feature_idx])
            if y == -1:
                return self._fit_data(x, branches[1])
            else:
                return self._fit_data(x, branches[2])
        else:
            # 遞迴到最後回傳一個數
            return branches[0]

    def _divide_data(self, xs, feature_idx, ys, s, theta):
        left = [[], []]
        right = [[], []]
        for idx, y in enumerate(ys):
            g = self._decision_stump(s, theta, xs[idx][feature_idx])
            if g == 1:
                list_added = right
            else:
                list_added = left

            x_added = list_added[0]
            x_added.append(xs[idx])
            y_added = list_added[1]
            y_added.append(y)
        return left, right

    def _branching(self, xs, ys):
        self._branch_count += 1
        s_s = [1, -1]
        best_s = 1,
        best_b = math.inf
        best_theta = math.inf
        best_feature = 0
        thetas = self._get_thetas(xs)
        for idx_theta_list, theta_list in enumerate(thetas):
            for s in s_s:
                for theta in theta_list:
                    # 用某feature 放入 演算法之後分左邊跟右邊
                    # 分完之後的list 位置分別為(x, y)
                    left_data, right_data = self._divide_data(xs, idx_theta_list, ys, s, theta)
                    # 分完之後拿g去看分的狀況
                    left_gini = self._compute_gini_index(left_data[1])
                    right_gini = self._compute_gini_index(right_data[1])
                    # b = sum_c(d_c * gini_b)
                    # c : 分支, d_c : 分支的資料數, gini_b : 分支的不純度
                    b = len(left_data[0]) * left_gini + len(right_data[0]) * right_gini
                    if b < best_b:
                        # 找到最純的分之
                        best_b = b
                        best_s = s
                        best_theta = theta
                        best_feature = idx_theta_list

        # 找到最純的之後，再將資料分類
        left_data, right_data = self._divide_data(xs, best_feature, ys, best_s, best_theta)

        return [best_s, best_theta, best_feature], left_data, right_data

    def build_tree(self):
        self._branch_count = 0
        self._branches = self._create_branch(self._x_data, self._y_data)

    def predict(self, x_s):
        result = []
        for x in x_s:
            predict_y = self._fit_data(x, self._branches)
            result.append(predict_y)
        return result

    def get_branch_count(self):
        return self._branch_count