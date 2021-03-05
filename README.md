# Predikon
_Sub-Matrix Factorization for Real-Time Vote Prediction_

[![Build Status](https://travis-ci.com/indy-lab/predikon-lib.svg?branch=master)](https://travis-ci.com/indy-lab/predikon-lib)
[![Coverage Status](https://coveralls.io/repos/github/indy-lab/predikon/badge.svg?branch=master)](https://coveralls.io/github/indy-lab/predikon?branch=master)
[![PyPI](https://img.shields.io/pypi/v/predikon?color=blue)](https://pypi.org/project/predikon/)
[![Downloads](https://pepy.tech/badge/predikon)](https://pepy.tech/project/predikon)

The `predikon` library is the Python library for the algorithm proposed in

> Alexander Immer\*, Victor Kristof\*, Matthias Grossglauser, Patrick Thiran, [*Sub-Matrix Factorization for Real-Time Vote Prediction*](https://infoscience.epfl.ch/record/278872), KDD 2020

The `predikon` algorithm enables you to predict aggregate vote outcomes (e.g., national) from partial outcomes (e.g., regional) that are revealed sequentially.
See the [usage documentation](docs/usage.md) more details on how to use this library or read the paper linked above for more details on how the algorithm works.

It is the algorithm powering [predikon.ch](http://www.predikon.ch), a platform for real-time vote prediction in Switzerland.

## Installation

To install the Predikon library from PyPI, run

```bash
pip install predikon
```

## Getting Started

Given a dataset `Y` of historical vote results collected in an array of `R` regions and `V` votes, given a vector `y` of partial results, and given an optional weighting `w` per region (e.g., the number of valid votes in each region), it is easy to predict the unobserved entries of `y` after observing at least one regional result (one entry of `y`) of an ongoing referendum or election:

```python
from predikon import LogisticSubSVD
model = LogisticSubSVD(Y, w)
pred = model.fit_predict(y)
# All unobserved entries in `y` are now filled in.
```

You can then obtain a prediction for the aggregate outcome (assuming the weights are the number of valid votes in this example) as:

```python
N = w.sum()  # Total number of votes.
ypred = pred.dot(w) / N
ytrue = y.dot(w) / N
print(abs(ypred - ytrue))
```

Have a look at the [example notebook](notebooks/example.ipynb) for a complete example of how to use the `predikon` library (with Swiss referenda).

## Going Further

You can find further information in:

- The [example notebook](notebooks/example.ipynb) using Swiss referenda
- The [usage documentation](docs/usage.md) describing the set up in more details
- The [scientific paper](https://infoscience.epfl.ch/record/278872) introducing the algorithm

And don't hesitate to **reach out us** if you have questions or to **open issues**!

## Requirements

- Python 3.5 and above
- [NumPy](https://numpy.org) 1.0.0 and above
- [scikit-learn](https://scikit-learn.org) 0.16.1 and above
