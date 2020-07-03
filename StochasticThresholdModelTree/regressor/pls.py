from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


class PLS(BaseEstimator, RegressorMixin):
    def __init__(self, n_components=None, do_optimize=True, n_splits=5):
        self.n_components = n_components
        self.do_optimize = do_optimize
        self.n_splits = n_splits

    def fit(self, X, y):
        if self.do_optimize:
            self.optimize_hyperparameter(X, y, random_state=0)
        if self.n_components is None:
            print('error')
            return

        y = y.reshape(-1, 1)

        self.W = np.zeros((X.shape[1], self.n_components))
        self.T = np.zeros((X.shape[0], self.n_components))
        self.P = np.zeros((X.shape[1], self.n_components))
        self.Q = np.zeros((y.shape[1], self.n_components))
        self.B = np.zeros((X.shape[1], self.n_components))
        for ii in range(self.n_components):
            self.W[:, [ii]] = np.dot(X.T, y)/np.linalg.norm(np.dot(X.T, y))
            self.T[:, [ii]] = np.dot(X, self.W[:, [ii]])
            self.P[:, [ii]] = np.dot(X.T, self.T[:, [ii]])/np.linalg.norm(
                np.dot(self.T[:, [ii]].T, self.T[:, [ii]]))
            self.Q[:, [ii]] = np.dot(y.T, self.T[:, [ii]])/np.linalg.norm(
                np.dot(self.T[:, [ii]].T, self.T[:, [ii]]))
            self.B[:, [ii]] = np.dot(np.dot(self.W[:, :ii+1], np.linalg.inv(
                np.dot(self.P[:, :ii+1].T, self.W[:, :ii+1]))),
                self.Q[:, :ii+1].T)
            X = X - np.dot(self.T[:, [ii]], self.P[:, [ii]].T)
            y = y - np.dot(self.T[:, [ii]], self.Q[:, [ii]])

    def predict(self, X):
        return np.dot(X, self.B[:, self.n_components-1])

    def predict_for_all_componets(self, X):
        return np.dot(X, self.B)

    def get_params(self, deep=True):
        return {
            'n_components': self.n_components,
            'do_optimize': self.do_optimize,
            'n_splits': self.n_splits}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def optimize_hyperparameter(self, X, y, random_state=None):
        y = y.reshape(-1, 1)
        max_n_components = np.linalg.matrix_rank(X)
        y_pred = np.zeros((X.shape[0], max_n_components))
        for index_train, index_test in KFold(n_splits=self.n_splits,
                                             shuffle=True,
                                             random_state=random_state
                                             ).split(X):
            X_train = X[index_train, :]
            y_train = y[index_train, :]
            X_test = X[index_test, :]
            model = PLS(n_components=max_n_components, do_optimize=False)
            model.fit(X_train, y_train)
            y_pred[index_test] = model.predict_for_all_componets(X_test)
        r2 = []
        for ii in range(max_n_components):
            common_index = (~np.isnan(y_pred[:, [ii]]) & ~np.isnan(y)).ravel()
            r2.append(r2_score(y[common_index], y_pred[common_index, ii]))
        self.n_components = np.argmax(r2) + 1


if __name__ == '__main__':
    X = np.array([
        [2, 2],
        [1, -1],
        [-1, 1],
        [-2, -2]
    ])
    y = np.array([
        [2],
        [0.5],
        [-0.5],
        [-2]
    ])

    model = PLS(n_splits=3)
    model.fit(X, y)
    print(model.n_components)
    print(model.B)
    print(model.B[:, model.n_components-1])
