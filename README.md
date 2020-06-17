# Predikon: online prediction of election results
[![Build Status](https://travis-ci.com/indy-lab/predikon.svg?branch=master)](https://travis-ci.com/indy-lab/predikon)
[![Coverage Status](https://coveralls.io/repos/github/indy-lab/predikon/badge.svg?branch=master)](https://coveralls.io/github/indy-lab/predikon?branch=master)
## Installation

```bash
pip install -r requirements.txt
pip install .
```

## Usage

We are given a `Region x Votes` matrix `M_historical` of historical results where each entry is in `[0,1]` indicating the fraction of _yes_ and _no_ votes for a binary outcome election.
Further, we have n optional weighting parameter with weights `weighting`, a vector with `Region` entries, indicating how many people voted in a particular region.
For an ongoing election, we have `m_current`.
`m_current` is a vector with `Region` entries containing observed and unobserved regions of an ongoing vote.
Observed entries are in `[0,1]` while unobserved entries are `np.nan` or `None`.


```python
from predikon import LogisticSubSVD
model = LogisticSubSVD(M_historical, weighting)
m_predict = model.fit_predict(m_current)
# all unknowns in `m_predict` are filled now
```



