import numpy as np


class MidPoint():
    """Class for selecting the midpoints for a candidate for the threshold."""
    def __init__(self):
        pass

    def __call__(self, start, end, seed):
        return (start + end) / 2


class NormalGaussianDistribution():
    """Class for stochastically determining candidate for the threshold using 
    a normal distribution.
    """
    def __init__(self, n_sigma):
        self.n_sigma = n_sigma

    def __call__(self, start, end, seed):
        if seed is not None:
            np.random.seed(seed)
        return np.random.normal(
            loc=(start + end) / 2,
            scale=(end - start) / (2 * self.n_sigma)
        )


if __name__ == '__main__':
    vals = [-5, -4, -3, -2, -1, 0, 1, 2]

    selector = MidPoint()
    thresholds = []
    start = vals[0]
    for end in vals[1:]:
        thresholds.append(selector(start, end))
        start = end
    print(thresholds)

    selector = NormalGaussianDistribution(1)
    thresholds = []
    start = vals[0]
    for end in vals[1:]:
        thresholds.append(selector(start, end, seed=None))
        start = end
    print(thresholds)

    selector = NormalGaussianDistribution(5)
    thresholds = []
    start = vals[0]
    for end in vals[1:]:
        thresholds.append(selector(start, end, seed=None))
        start = end
    print(thresholds)
