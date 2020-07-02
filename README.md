# Ensemble-trees
===

Random Forest is a highly flexible and accurate machine learning method and is used in many situations. However, its nature makes it unsuitable for extrapolation prediction.
The proposed method can provide reasonable extrapolation predictions for physicochemical and other data that are expected to have a certain degree of monotonicity.

## Installation

You can install the repository into your local environment by the following command.

```bash
$ pip install git+https://github.com/funatsu-lab/Ensemble-trees.git
```

## Usage

The module is imported and used as follows.

```python
from regressor.EnsembleTrees import EnsembleTrees
from threshold_selector import NormalGaussianDistribution
from criterion import MSE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
model = EnsembleTrees(n_estimators=100, # The number of regression trees to create
                      criterion=MSE(), # Criteria for setting divisional boundaries
                      regressor=LinearRegression(), # Regression model applied to the sample group of each terminal node
                      threshold_selector=NormalGaussianDistribution(5), # Parameters for determining the candidate division boundary
                      min_samples_leaf=1.0, # Minimum number of samples required to make up a node
                      max_features='auto', # Number of features to consider for optimal splitting
                      f_select=True, # If true, the regression is performed using only the features used for splitting in each regression tree.
                      ensemble_pred='median', # During the ensemble, whether to take the mean or the median
                      scaling=False, # Whether to perform standardization as a pre-processing when applying linear regression to each terminal node
                      random_state=None
                      )
X = 
y = 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test) # output of the embedding
```
