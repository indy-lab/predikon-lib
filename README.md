# Predikon: online prediction of election results
[![Build Status](https://travis-ci.com/indy-lab/predikon.svg?branch=master)](https://travis-ci.com/indy-lab/predikon)
[![Coverage Status](https://coveralls.io/repos/github/indy-lab/predikon/badge.svg?branch=master)](https://coveralls.io/github/indy-lab/predikon?branch=master)

## Usage

![problem_structure](https://user-images.githubusercontent.com/7715036/86534364-346d1980-bed8-11ea-8956-439354c87285.png)


#### Historical outcomes
As detailed above, we are dealing with a matrix or tensor
<img src="https://render.githubusercontent.com/render/math?math=Y_V">
of fully observed vote outcomes.
This object contains the historical outcomes and will be used to infer historical
voting patterns.
In case of a binary outcome, as for example in referenda,
we work with a matrix and the entry corresponding to the
r-th region and v-th vote gives us the fraction of positive outcomes, e.g., fraction of _yes_ in the
case of referenda.
Alternatively, in parliamentary elections we require a vector
description of outcomes that sums up to one.
Assuming we have K parties or options, the resulting outcome
tensor will be of shape R x V x K.
In the code, the matrix
<img src="https://render.githubusercontent.com/render/math?math=Y_V">
is denoted by `M_historical` and is used to initialize models.

#### Weighting of regions
Additionally, we support to weight regions differently.
Typically, the weighting will be the number of valid votes for a particular region.
Alternatively, one can work with approximate values of this number, e.g., the number of people
living in each region.
The weighting vector is called `weighting` and is required at model initialization together
with the historical outcomes.
If no such data is available, one can pass `weighting=None` to ignore weighting.


#### Online prediction of outcomes
To predict the outcomes of an ongoing election, unobserved entries of the
partially observed vector or matrix
<img src="https://render.githubusercontent.com/render/math?math=y_{V %2B 1}">
need to be inferred.
For binary outcomes, we have an R dimensional vector and for multiple options we have a R x K
dimensional matrix where observed entries are summed to one again.
To infer these using one of the methods implemented, the partially observed vector or matrix
`m_current` contains `np.nan` (or equivalently None) entries for unobserved entries.
These unobserved entries are then filled in by a method of choice by calling
`model.fit_predict(m_current)`.

#### Putting it together
Assuming you have the above quantities ready for a particular election (`M_historical, weighting, m_current`),
inferring unobserved entries after observing at least one regional outcome of an ongoing election is
easy:
```python
from predikon import LogisticSubSVD
model = LogisticSubSVD(M_historical, weighting)
m_predict = model.fit_predict(m_current)
# all unknowns in `m_predict` are filled now
```

#### Choosing the right model
For binary outcomes, i.e., if we have an outcome matrix, the preferred method is `LogisticSubSVD`.
For multiple possible outcomes, i.e., the tensor case, `LogisticTensorSubSVD` is preferred.
Both further have two important parameters `l2_reg` and `n_dim`:
`l2_reg` controls the strength of l2 regularization of the parameters while
`n_dim` controls the number of latent dimensions used for the model.
In many cases, not much historical data is available and thus the number of dimensions
can be kept low, for example, set between 5 and 10.
The `l2_reg` parameter defaults to `1e-5` but should be tuned for a particular problem, e.g.,
using cross-validation.

Alternatively, the standard (weighted) averaging (`WeightedAveraging`) baseline is also available as well as
Gaussian versions of SubSVD model (`GaussianSubSVD` and `GaussianTensorSubSVD`).
Further, for the binary outcome probabilistic matrix factorisation (`MatrixFactorisation`) is implemented.


## Installation

```bash
pip install -r requirements.txt
pip install .
```

