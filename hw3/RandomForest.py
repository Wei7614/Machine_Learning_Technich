import math
import random
from Cart import Cart


# random forest 就是 bagging(boostrapping) + decision tree
class RandomForest:

    trees = []
    # private variable
    _x_data = None
    _y_data = None

    def __init__(self, x, y):
        self._x_data = x
        self._y_data = y

    def _boostrap(self, x, y, N):
        indies = [random.randint(0, N) for _ in range(N)]
        x_result = []
        y_result = []
        for idx in indies:
            x_result.append(x[idx])
            y_result.append(y[idx])
        return x_result, y_result

    def _fit(self, x):
        result = []
        for t in self.trees:
            # 這筆data在t這棵樹所做出來的預測
            predict_y = t.predict([x])
            result.append(predict_y[0])

        # 每棵樹做完的結果之後綜合起來然後回傳占比最大的值
        if sum(result) >= 0:
            return 1
        return -1

    def build_trees(self, t, is_prune=False):
        self.trees = []
        for i in range(t):
            bx, by = self._boostrap(self._x_data, self._y_data, len(self._y_data) - 1)
            cart = Cart(bx, by)
            cart.is_prune = is_prune
            cart.build_tree()
            self.trees.append(cart)

    def predict(self, x):
        result = []
        # 用data_n 去fit每ㄧ顆數
        for n in range(len(x)):
            predict_y = self._fit(x[n])
            result.append(predict_y)
        return result





