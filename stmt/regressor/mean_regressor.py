import numpy as np


class MeanRegressor():
    """Class for outputting the mean value of each node as a prediction.
    The predicted results are comparable to those of Random Forest.
    """
    def __init__(self):
        pass

    def fit(self, x, y):
        self.mean = np.mean(y)

    def predict(self, x):
        return np.full(x.shape[0], self.mean)
