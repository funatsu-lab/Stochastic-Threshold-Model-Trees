import numpy as np
import copy
from sklearn.utils import check_array, check_X_y
from sklearn.preprocessing import StandardScaler
from ..criterion import MSE
from ..criterion import MSE_by_model
from ..threshold_selector import MidPoint
from ..threshold_selector import NormalGaussianDistribution


class Node():
    """Class for actual model constructing."""
    def __init__(
        self,
        criterion=None,
        regressor=None,
        threshold_selector=None,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        f_select=True,
        scaling=True,
        random_state=None,
        selected_features=set(),
        split_continue=False,
        original_X=None,
        coef_matrix=None,
        model_flag=False
    ):
        self.criterion = criterion
        self.regressor = regressor
        self.threshold_selector = threshold_selector
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.f_select = f_select
        self.scaling = scaling
        self.random_state = random_state
        self.selected_features = selected_features
        self.split_continue = split_continue
        self.original_X = original_X
        self.coef_matrix = coef_matrix
        self.model_flag = model_flag
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.information_gain = None
        self.scalerX = None
        self.scalerY = None

    def construct_model(self, X, y):
        self.selected_features = sorted(self.selected_features)
        if self.f_select:
            X = X[:, self.selected_features]
        else:
            X = X

        if self.scaling:
            self.scalerX = StandardScaler()
            X_std = self.scalerX.fit_transform(X)
            self.scalerY = StandardScaler()
            y_std = self.scalerY.fit_transform(y.reshape(-1, 1))
            self.regressor.fit(X_std, y_std)
        else:
            self.regressor.fit(X, y)

        if self.split_continue:
            n_samples = self.original_X.shape[0]
            n_features = X.shape[1]
            if self.coef_matrix is None:
                self.coef_matrix = np.zeros((n_samples, n_features))
            original_X = self.original_X[:, self.selected_features]
            node_index = []
            for node_sample in X:
                index = [sum(l) for l in original_X - node_sample].index(0)
                node_index.append(index)
            self.coef_matrix[node_index] = self.regressor.coef_
            self.model_flag = True

    def reconstruct_model(self, X, y):
        original_X = self.original_X[:, list(self.selected_features)]
        node_index = []
        for node_sample in X:
            index = [sum(l) for l in original_X - node_sample].index(0)
            node_index.append(index)
        self.regressor.intercept_ = y - np.dot(X, self.coef_matrix[node_index].T)

    def fit(self, X, y, depth):
        X, y = check_X_y(X, y, ['csr', 'csc'])

        if depth == 0:
            self.original_X = X

        n_samples, n_features = X.shape
        # random_state = self.check_random_state(self.random_state)
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Define the minimum number of samples at each node.
        if self.min_samples_leaf == 'random':
            if self.f_select:
                min_samples_leaf = int(np.ceil(
                    (np.random.rand() + 1.) * max(5, len(self.selected_features))))
            else:
                min_samples_leaf = int(np.ceil(
                    (np.random.rand() + 1.) * n_features))

        elif isinstance(self.min_samples_leaf, int):
            if not 1 <= self.min_samples_leaf:
                print('The number of samples at each node must be one or more.')
            min_samples_leaf = self.min_samples_leaf
        else:
            if self.f_select:
                min_samples_leaf = int(np.ceil(
                    self.min_samples_leaf * len(self.selected_features)))
            else:
                min_samples_leaf = int(np.ceil(
                    self.min_samples_leaf * n_features))
            min_samples_leaf = max(1, min_samples_leaf)

        if isinstance(self.min_samples_split, int):
            if not 2 <= self.min_samples_split:
                print('The number of samples at each node before split must be two or more.')
            min_samples_split = self.min_samples_split
        else:
            if self.f_select:
                min_samples_split = int(np.ceil(
                    self.min_samples_split * len(self.selected_features)))
            else:
                min_samples_split = int(np.ceil(
                    self.min_samples_split * n_features))
            min_samples_split = max(2, min_samples_split)

        min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

        # If the split end condition is satisfied, construct the model.
        if depth == self.max_depth:
            self.construct_model(X, y)
            return
        if n_samples <= min_samples_split:
            if not self.model_flag:
                self.construct_model(X, y)
            if self.split_continue:
                min_samples_leaf = 1  # It can be said that this should be a hyper parameter.
                if n_samples <= min_samples_leaf:
                    self.reconstruct_model(X, y)
                    return
            else:
                return

        self.information_gain = 0.0
        if self.random_state is not None:
            np.random.seed(self.random_state)
        for feature in range(n_features):
            uniq_feature = np.unique(X[:, feature])
            if min_samples_leaf > 1:
                uniq_feature = uniq_feature[
                    min_samples_leaf - 1:min_samples_leaf * -1 + 1]
            seeds = np.random.randint(
                np.iinfo(np.int32).max, size=len(uniq_feature) + 1)
            thresholds = [
                self.threshold_selector(start, end, seed=seed)
                for start, end, seed
                in zip(uniq_feature[:-1], uniq_feature[1:], seeds)]
            for threshold in thresholds:
                index_left = (X[:, feature] <= threshold)
                index_right = (X[:, feature] > threshold)
                information_gain = self.criterion(
                    X, X[index_left], X[index_right],
                    y, y[index_left], y[index_right])
                if self.information_gain < information_gain:
                    self.information_gain = information_gain
                    self.feature = feature
                    self.threshold = threshold

        if self.information_gain == 0.0:
            self.construct_model(X, y)
            return

        self.left = Node(
            criterion=self.criterion,
            regressor=copy.deepcopy(self.regressor),
            threshold_selector=self.threshold_selector,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            f_select=self.f_select,
            scaling=self.scaling,
            random_state=self.random_state,
            selected_features=set(self.selected_features).union({self.feature}),
            split_continue=self.split_continue,
            original_X=self.original_X,
            coef_matrix=self.coef_matrix,
            model_flag=self.model_flag
            )

        self.right = Node(
            criterion=self.criterion,
            regressor=copy.deepcopy(self.regressor),
            threshold_selector=self.threshold_selector,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            f_select=self.f_select,
            scaling=self.scaling,
            random_state=self.random_state,
            selected_features=set(self.selected_features).union({self.feature}),
            split_continue=self.split_continue,
            original_X=self.original_X,
            coef_matrix=self.coef_matrix,
            model_flag=self.model_flag
            )

        index_left = (X[:, self.feature] <= self.threshold)
        index_right = (X[:, self.feature] > self.threshold)

        self.left.fit(X[index_left], y[index_left], depth + 1)
        self.right.fit(X[index_right], y[index_right], depth + 1)

    def predict(self, X):
        X = check_array(X, accept_sparse='csr')

        n_samples = X.shape[0]
        if (self.right is None) or (self.left is None):
            if self.f_select:
                X = X[:, self.selected_features]
            else:
                X = X

            if self.scaling:
                X_std = self.scalerX.transform(X)
                y_pred_std = self.regressor.predict(X_std).reshape(-1)
                return self.scalerY.inverse_transform(y_pred_std)
            else:
                return self.regressor.predict(X).reshape(-1)
        else:
            pred = np.zeros(n_samples)
            is_left = (X[:, self.feature] <= self.threshold)
            is_right = (np.ones(n_samples, dtype=bool))
            is_right[is_left] = False
            if sum(is_left) != 0:
                pred[is_left] = self.left.predict(X[is_left, :])
            if sum(is_right) != 0:
                pred[is_right] = self.right.predict(X[is_right, :])
            return pred

    def _count_feature(self):
        if (self.right is None) or (self.left is None):
            return len(self.selected_features)
        else:
            left_feature = self.left._count_feature()
            right_feature = self.right._count_feature()
            return max(left_feature, right_feature)

    def check_random_state(self, seed):
        if seed is None or seed is np.random:
            return np.random.mtrand._rand
        if isinstance(seed, int):
            return np.random.RandomState(seed)
        if isinstance(seed, np.random.RandomState):
            return seed
