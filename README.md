# Stochastic Threshold Model Trees

Stochastic Threshold Model Trees provides reasonable extrapolation predictions for physicochemical and other data that are expected to have a certain degree of monotonicity.

## Requirements
- Python 3
- [Numpy](https://numpy.org/) >= 1.17
- [joblib](https://pypi.org/project/joblib/) == 0.13
- [scikit-learn](https://scikit-learn.org/stable/) == 0.21

## Requirements for notebook
- [pandas](https://pandas.pydata.org/) >= 0.25
- [matplotlib](https://matplotlib.org/) >= 3.1
- [seaborn](https://seaborn.pydata.org/) >= 0.9

## Installation

You can install the repository into your local environment by the following command.

```bash
$ pip install git+https://github.com/funatsu-lab/Stochastic-Threshold-Model-Trees.git
```

## Examples

As shown in the figure below, the proposed method makes predictions that reflect the trend of the sample near the extrapolation area.

![discontinuous_Proposed_5sigma](https://user-images.githubusercontent.com/49966285/86465964-ad039700-bd6d-11ea-80b0-8035fc726228.png)

![Sphere_Proposed_MLR_noise_scaling](https://user-images.githubusercontent.com/49966285/86466391-7d08c380-bd6e-11ea-879c-8e9b3f9ba493.png)

![1dim_comparison](https://user-images.githubusercontent.com/49966285/86992420-69c97e00-c1dc-11ea-8e2f-8b3d08ce27d4.png)


## Usage

The module is imported and used as follows.

```python
from StochasticThresholdModelTrees.regressor.stmt import StochasticThresholdModelTrees
from StochasticThresholdModelTrees.threshold_selector import NormalGaussianDistribution
from StochasticThresholdModelTrees.criterion import MSE
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model = StochasticThresholdModelTrees(
  n_estimators=100, # The number of regression trees to create
  criterion=MSE(), # Criteria for setting divisional boundaries
  regressor=LinearRegression(), # Regression model applied to each terminal node
  threshold_selector=NormalGaussianDistribution(5), # Parameters for determining the candidate division boundary
  min_samples_leaf=1.0, # Minimum number of samples required to make up a node
  max_features='auto', # Number of features to consider for optimal splitting
  f_select=True, # Whether or not to choose features to consider when splitting
  ensemble_pred='median', # During the ensemble, whether to take the mean or the median
  scaling=False, # Whether to perform standardization as a pre-processing to each terminal node
  random_state=None
  )
data = pd.read_csv('./data/logSdataset1290.csv', index_col=0)
X = data[data.columns[1:]]
y = data[data.columns[0]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) # Model predictions
```

## Reference:
[Stochastic Threshold Model Trees: A Tree-Based Ensemble Method for Dealing with Extrapolation](https://arxiv.org/abs/2009.09171)

## License
MIT