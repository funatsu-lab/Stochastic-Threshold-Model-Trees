import numpy as np


class MeanRegressor():
    def __init__(self):
        pass

    def fit(self, x, y):
        self.mean = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self.mean)
