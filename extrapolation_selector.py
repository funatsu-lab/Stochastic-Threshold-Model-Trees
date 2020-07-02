import numpy as np
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import StandardScaler


class selector_based_y():
    def __init__(self, fraction):
        self.fraction = fraction

    def fit(self, X, y):
        X, y = check_X_y(X, y, ['csr', 'csc'])
        self.index = np.argsort(y.flatten())

    def predict(self, X):
        X = check_array(X, accept_sparse='csr')
        n = X.shape[0]
        self.score = np.zeros(n)
        threshold = int(np.floor(n * self.fraction))
        extrapolation = self.index[:threshold]
        extrapolation = np.hstack((extrapolation, self.index[-threshold:]))
        interpolation = self.index[threshold:-threshold]
        self.score[extrapolation] = -1
        self.score[interpolation] = 1
        return self.score


class selector_based_X():
    def __init__(self, fraction, criterion='median'):
        self.fraction = fraction
        self.criterion = criterion

    def fit(self, X):
        X = check_array(X, accept_sparse='csr')
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        if self.criterion == 'median':
            center = np.median(X_std, axis=0)
        else:
            center = np.mean(X_std, axis=0)

        self.dist = [np.linalg.norm(x - center) for x in X_std]
        self.index = np.argsort(self.dist)

    def predict(self, X, return_dist=False):
        X = check_array(X, accept_sparse='csr')
        n = X.shape[0]
        self.score = np.zeros(n)
        threshold = int(np.floor(n * self.fraction * 2.))
        extrapolation = self.index[-threshold:]
        interpolation = self.index[:-threshold]
        self.score[extrapolation] = -1
        self.score[interpolation] = 1

        if return_dist:
            return self.score, self.dist
        else:
            return self.score


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axis3d

    def my_plot(X1_train, X2_train, y_train, X1_out=None, X2_out=None, y_out=None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['axes.linewidth'] = 1.0

        ax.scatter(X1_train, X2_train, y_train, s=30, c='b', marker='o')
        min_x1 = X1_train.min()
        max_x1 = X1_train.max()
        min_x2 = X2_train.min()
        max_x2 = X2_train.max()
        min_y = y_train.min()
        max_y = y_train.max()

        if X1_out is not None and X2_out is not None and y_out is not None:
            ax.scatter(X1_out, X2_out, y_out, s=2, c='r', marker=',')
            min_x1 = min(min_x1, X1_out.min())
            max_x1 = max(max_x1, X1_out.max())
            min_x2 = min(min_x2, X2_out.min())
            max_x2 = max(max_x2, X2_out.max())
            min_y = min(min_y, y_out.min())
            max_y = max(max_y, y_out.max())

        ax.set_xlim(min_x1 - 0.05 * (max_x1 - min_x1), max_x1 + 0.05 * (max_x1 - min_x1))
        ax.set_ylim(min_x2 - 0.05 * (max_x2 - min_x2), max_x2 + 0.05 * (max_x2 - min_x2))
        ax.set_zlim(min_y - 0.05 * (max_y - min_y), max_y + 0.05 * (max_y - min_y))

        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_zlabel(r"$y$")

        plt.tight_layout()
        plt.show()

    def sphere(X1, X2):
        return X1 ** 2 + X2 ** 2

    def goldstein(X1, X2):
        return (1+(X1+X2+1)**2*(19-14*X1+3*X1**2-14*X2+6*X1*X2+3*X2**2))*(30+(2*X1-3*X2))**2*(18-32*X1+12*X1**2+48*X2-36*X1*X2+27*X2**2)

    x1_train = np.arange(-5, 5.2, 0.2)
    x2_train = np.arange(-5, 5.2, 0.2)

    X1_train, X2_train = np.meshgrid(x1_train, x2_train)
    y_train = sphere(X1_train, X2_train)
    np.random.seed(0)
    X1_train += np.random.randn(X1_train.shape[0], X1_train.shape[1]) * 0.1
    X2_train += np.random.randn(X2_train.shape[0], X2_train.shape[1]) * 0.1
    y_train += np.random.randn(y_train.shape[0]) * 0.5
    y_train = y_train.flatten()

    X_train = np.array([[X1, X2] for X1 in x1_train for X2 in x2_train])

    selector = selector_based_X(fraction=0.1)
    selector.fit(X_train, y_train)
    score = selector.predict(X_train)
    X_out = X_train[score < 0]
    X1_out = X_out[:, 0]
    X2_out = X_out[:, 1]
    y_out = y_train.flatten()[score < 0]
    X_in = X_train[score > 0]
    X1_in = X_in[:, 0]
    X2_in = X_in[:, 1]
    y_in = y_train.flatten()[score > 0]
    my_plot(X1_in, X2_in, y_in, X1_out, X2_out, y_out)
