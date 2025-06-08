import pandas as pd
import numpy as np
from scipy.stats import norm

def probit(x):
    return [norm.cdf(x), norm.pdf(x), -x * norm.pdf(x)]
def logit(x):
    y = 1 / (1 + np.exp(-x))
    y2 = np.exp(-x) / (1 + np.exp(-x)) ** 2
    y3 = (np.exp(-2 * x) - np.exp(-x)) / (1 + np.exp(-x)) ** 3
    return [y, y2, y3]
def bhood(func, param, y, x):
    param = param.reshape(-1, 1)
    z = np.dot(x, param)
    g = func(z)
    u = y * np.log(g[0]) + (1 - y) * np.log(1 - g[0])
    u2 = np.sum(u, axis=0)
    v = (y * (g[1] / g[0])) * x - ((1 - y) * g[1] / (1 - g[0])) * x
    v2 = np.sum(v, axis=0)
    h = (g[0] * g[2] - g[1] ** 2) / g[0] ** 2
    h2 = (-(1 - g[0]) * g[2] - g[1] ** 2) / (1 - g[0]) ** 2
    w = np.dot(x.T, y * h * x) + np.dot(x.T, (1 - y) * h2 * x)
    return [u2, v2, w]

class binary_regression:
    def __init__(self, mode="logit", tol=1e-4):
        """
        :param mode: "logit" or "probit"
        :param tol: Newton法の閾値
        """
        self.mode = mode
        self.tol = tol

        self.coef = None
        self.se = None
        self.t = None
        self.APE = None
        self.PEA = None
        self.likelihood = None
        self.n = None
        self.k = None

    def _get_func(self):
        """
        function を mode に応じて取得
        :return: func
        """
        if self.mode == "logit":
            return logit
        elif self.mode == "probit":
            return probit
        else:
            raise NotImplementedError

    def _get_z_y(self, x, y=None):
        """
        - 説明変数 x, 目的変数 yの変数名の取得 (array形式の時は数字で代替)
        - 切片項の追加
        - array形式に変更

        :param x: 説明変数
        :param y: 目的変数 (0 or 1)
        :return: z (const+x), y
        """
        self.n, self.k = x.shape  # get N & K
        const = np.ones((self.n, 1))  # make constant val

        if not isinstance(x, np.ndarray):  # get x names
            self.x_names = ["const"] + x.columns.tolist()
        else:
            self.x_names = [f"x{i:02}" for i in range(self.k + 1)]

        z = np.hstack([const, np.array(x)])  # const. + x
        if y is None:  # y がなかったらzだけ返す
            return z

        if not isinstance(y, np.ndarray):  # get y name
            self.y_name = y.name if isinstance(y, pd.Series) else y.columns[0]
        else:
            self.y_name = "y"
        y = np.array(y)

        return z, y

    def fit(self, x, y):
        """
        Newton法を用いた推定(mle funcとほぼ同じ)
        メンバ変数に格納(coef, se, t, spe, pea, etc.)
        :param x: 目的変数
        :param y: 説明変数
        :return: None
        """
        z, y = self._get_z_y(x, y)  # get z & y
        func = self._get_func()  # get func

        param = np.zeros(z.shape[1])
        f = bhood(func, param, y, z)
        r = max(abs(f[1]))

        while r > self.tol:
            param = param - np.dot(np.linalg.inv(f[2]), f[1])
            f = bhood(func, param, y, z)
            r = max(abs(f[1]))

        self.coef = param
        self.coef_= param[1:]
        self.intercept_ = param[0]
        self.likelihood = f[0]
        self.se = np.sqrt(np.diag(np.linalg.inv(-f[2])))
        self.t = self.coef / self.se
        self.p = (1 - norm.cdf(abs(self.t)))*2
        u = func(np.dot(z, param.reshape(-1, 1)))
        self.APE = np.mean(u[1]) * param
        u = func(np.dot(np.mean(z, axis=0), param))
        self.PEA = u[1] * param
        return self

    def predict(self, x):
        """
        引数xとfitで推定したパラメータをもとに 1である予測確率を返す
        :param x: 説明変数
        :return: 予測確率(0~1)
        """
        if x.shape[1] != self.k:
            raise ValueError

        z = self._get_z_y(x)
        z = z @ self.coef
        func = self._get_func()
        return func(z)[0]

    def summary(self):
        """
        summaryの出力
        :return: summary dataframe
        """
        col_names = ["coef", "se", "t", "p","PEA", "APE"]
        output = pd.DataFrame(np.c_[self.coef, self.se, self.t, self.p, self.PEA, self.APE],
                              index=self.x_names,
                              columns=col_names)
        return output
