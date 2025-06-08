#sk-learn風の線形回帰分析クラス
#SE, t-val,p-val, R2を出力
#x, yをpandas DataFrameで入力、self.olsは(coef, SE, t値, p値)のDataFrame
from scipy.stats import t
import numpy as np 
import pandas as pd

class linear_regression():
    def __init__(self, fit_intercept=True):
        self.const = fit_intercept

    def fit(self, x, y):
        # pandas DataFrame → numpy配列に変換
        if isinstance(x, pd.DataFrame):
            x_np = x.values
            x_cols = x.columns
        else:
            x_np = x
            x_cols = [f"x{i}" for i in range(x.shape[1])]
        
        # yがSeriesかDataFrameかに関係なく1列の配列にする
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_np = np.asarray(y).reshape(-1, 1)
        else:
            y_np = y.reshape(-1, 1)

        if self.const:
            ones = np.ones((x_np.shape[0], 1))
            z = np.concatenate([ones, x_np], axis=1)
            self.index = ['const'] + list(x_cols)
        else:
            z = x_np
            self.index = list(x_cols)

        self.phi = np.linalg.inv(z.T @ z) @ (z.T @ y_np)
        if self.const:
            self.intercept_ = self.phi[0, 0]
            self.coef_ = self.phi[1:].flatten()
        else:
            self.intercept_ = 'NA'
            self.coef_ = self.phi.flatten()

        u = y_np - z @ self.phi
        RSS = np.sum(u**2)
        TSS = np.sum((y_np - np.mean(y_np))**2)
        self.R2 = 1 - RSS / TSS
        self.s2 = RSS / (z.shape[0] - z.shape[1])
        self.SE = np.sqrt(self.s2 * np.diagonal(np.linalg.inv(z.T @ z)))
        self.t = self.phi.flatten() / self.SE
        self.p = (1 - t.cdf(np.abs(self.t), df=z.shape[0] - z.shape[1])) * 2

    def predict(self, x):
        if isinstance(x, pd.DataFrame):
            x_np = x.values
        else:
            x_np = x

        if self.const:
            z = np.concatenate([np.ones((x_np.shape[0], 1)), x_np], axis=1)
        else:
            z = x_np

        fcst = z @ self.phi
        return fcst.squeeze()

    def summary(self):
        col_names = ["coef", "se", "t", "両側p値"]
        output = pd.DataFrame(np.c_[self.phi.flatten(), self.SE, self.t, self.p],
                              index=self.index,
                              columns=col_names)
        print(f'決定係数R^2: {self.R2}')
        return output
