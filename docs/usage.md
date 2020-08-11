# Usage

![problem_structure](https://user-images.githubusercontent.com/7715036/86534364-346d1980-bed8-11ea-8956-439354c87285.png)


## Historical Votes

Given R regions and V votes, we store a dataset of historical votes in a vote matrix (or a tensor)
<img src="https://render.githubusercontent.com/render/math?math=Y_V">
consisting of the (fully) observed vote outcomes.
In the case of a binary outcome, e.g., a referendum, each entry of the vote corresponds to the fraction of "yes" in the r-th region and v-th vote.
Alternatively, in the case of multiple outcomes, e.g., an election, each entry of the vote tensor is a vector that corresponds to the fraction of votes for each party.
(We assume that the sum of the entries in each vector equals 1).
For K parties, the vote tensor will be of shape R x V x K.
In the code, the matrix
<img src="https://render.githubusercontent.com/render/math?math=Y_V">
is denoted by `M_historical` and is used to initialize models.

## Weighting of Regions

Additionally, we support some weighting to account for disparities between regions.
For example, the weighting may be the number of valid votes or the population in each region.
The weighting vector is called `weighting` and is required when initializing models, together with the historical outcomes.
If no such data is available, one can pass `weighting=None` to ignore weighting.

## Real-Time Predictions

To predict the outcomes of an ongoing election in real-time, the algorithm predicts the unobserved entries of the partially observed vector (or matrix)
<img src="https://render.githubusercontent.com/render/math?math=y_{V %2B 1}">
For binary outcomes, this is an R-dimensional vector, and this is an R x K matrix for multiple outcomes.
In the code, the partially observed vectors `m_current` contains both actual, observed values in entries where the regional was available and `np.nan` in entries for unobserved regions.
These unobserved entries are then filled in by the method of choice by calling
`model.fit_predict(m_current)`.

## Putting It Together

Assuming you have the above quantities ready for a particular vote (i.e., `M_historical`, `weighting`, and `m_current`), inferring the unobserved entries after observing at least one regional outcome of an ongoing election is easy:
```python
from predikon import LogisticSubSVD
model = LogisticSubSVD(M_historical, weighting)
m_predict = model.fit_predict(m_current)
# All unobserved entries in `m_current` are now filled in.
```

## Choosing the Right Model (Hyperparameters)

For binary outcomes, i.e., for referenda, the preferred method is `LogisticSubSVD`.
For multiple possible outcomes, i.e., for elections, the preferred method is `LogisticTensorSubSVD`.
Both further have two important hyperparameters, `l2_reg` and `n_dim`:
- `l2_reg` controls the strength of the L2 regularization of the parameters
- `n_dim` controls the number of latent dimensions used for the model

In many cases, not much historical data is available and thus the number of dimensions can be set to a low value, e.g., between 5 and 10.
The `l2_reg` parameter defaults to `1e-5` but should be tuned for the problem at hand, e.g., using cross-validation.
Gaussian versions of our model (`GaussianSubSVD` and `GaussianTensorSubSVD`) are also available, and they require the same hyperparameters.

We provide additionally two baselines for comparison and model selection.
The standard (weighted) averaging (`WeightedAveraging`) and probabilistic matrix factorization (`MatrixFactorisation`) for binary outcomes.
