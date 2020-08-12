import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression


LARGE_FLOAT = 1e9


class Model:
    """
    Base class for real-time predictive models for regional results.

    Generally, subclasses of this model can be used in the following way:
    ```
    import numpy as np
    R, V = 100, 10
    # M is Region x Votes or Region x Votes x Parties.
    M_historical = np.random.randn(R, V)
    m_current = np.random.randn(R)
    # TODO: set some nans etc.
    # Weighting is R dimensional associated with each region.
    weightings = np.abs(np.random.randn(R))
    m = Model(M_historical, weighting)
    pred = m.fit_predict(m_current)
    # Pred has filled nans with predictions.
    ```

    Attributes:
        M_historical: RxV or RxVxP dimensional np.array with outcomes in [0,1],
            where R is the number of regions, V the number of votes, and P the
            number of parties (non-binary outcome).
        weighting: R dimensional np.array array with weights associated to each
            region; if `weighting=None` the algorithm is unweighted (equivalent
            to np.ones(R)).
    """

    def __init__(self, M_historical, weighting=None):
        R = M_historical.shape[0]
        if weighting is None:
            weighting = np.ones(R)
        if weighting.ndim != 1:
            raise ValueError(
                'Weighting must be an np.array of dimension {}'.format(R))
        if len(weighting) != M_historical.shape[0]:
            raise ValueError(
                'Weighting must be an np.array of dimension {}'.format(R))
        self.M_historical = M_historical
        # Make weights sum to R.
        self.weighting = weighting / np.sum(weighting) * R

    def fit_predict(self, m_current):
        """Fit the model and predict the unobserved values.

        Fit the model to the current observations and predict the missing
        entries of `m_current` as indicated by the np.nans.

        Arguments:
            m_current : R or RxP dimensional np.array with outcomes in [0,1]
                and unobserved entries which are given as np.nan.
        """
        raise NotImplementedError

    def get_obs_ixs(self, m_current):
        """Returns the indices of observed and unobserved regions.

        Turn the array `m_current` into two index-lists of observed and
        unobserved regions by detecting unobserved nan-values.

        Arguments:
            m_current : R or RxP dimensional np.array with outcomes in [0,1]
                and unobserved entries which are given as np.nan
        """
        if m_current.ndim == 1:
            assert len(m_current) == self.M_historical.shape[0]
            # R - vector.
            unobserved = np.isnan(m_current)
        elif m_current.ndim == 2:
            R, P = m_current.shape
            assert (R == self.M_historical.shape[0] and
                    P == self.M_historical.shape[2])
            # R x P dim.
            unobserved = np.any(np.isnan(m_current), axis=-1)

        obs, unobs = np.where(~unobserved), np.where(unobserved)
        return obs, unobs

    def __repr__(self):
        return 'Predikon BaseModel'


class WeightedAveraging(Model):
    """Describes a weighted average baseline.

    Same as `Averaging` but additionally weighs the observed entries in
    `m_current` by the given weights.  This model is preferred over Averaging
    if population or counts of votes are available.
    """

    def fit_predict(self, m_current):
        """See base class."""
        obs, unobs = self.get_obs_ixs(m_current)
        wts = self.weighting[obs]
        N = wts.sum()
        if m_current.ndim == 2:
            pred = (m_current[obs] * wts.reshape(-1, 1)).sum(axis=0) / N
        else:
            pred = (m_current[obs] * wts).sum(axis=-1) / N
        pred_col = np.ones_like(m_current) * pred
        pred_col[obs] = m_current[obs]
        return pred_col

    def __repr__(self):
        return 'Weighted Averaging'


class MatrixFactorisation(Model):
    r"""Describes a matrix factorization model.

    Probabilistic matrix factorization model that implements
    alternating least-squares to minimize the loss
    ``L(U, V) = 1/2 \|M - UV^T\|_2^2 + lam_U/2 \|U\|_2^2 + lam_V/2 \|V\|_2^2,``
    where U, V contain the latent factors and M is the concatenation of
    `M_historical` and `m_current`. The matrices `U` and `V` are updated in an
    alternating fashion. For the updates, see `update_U` and `update_V`.
    Variables are initialized uniformly between 0 and 1/(n_dim+1) `U` is
    a `Region x (n_dim+1)` and `V` a `Votes x (n_dim+1)` matrix. The
    dimensionality is increased because we include a bias per vote as in _Etter
    et al (2016)_.

    Matrix factorization for regional vote prediction is proposed
    in _Online Collaborative Prediction of Regional Vote Results_
    by Etter et al., DSAA 2016_.

    Attributes:
        n_iter: Number of alternating iterations which determines how often
            `U`, and `V` are updated each.
        n_dim: Number of latent dimensions; does not include the bias.
        lam_U: Regularizer for latent factors `U`.
        lam_V: Regularizer for latent factors `V`.
    """

    def __init__(self,
                 M_historical,
                 weighting,
                 n_iter=20,
                 n_dim=25,
                 lam_U=1e-1,
                 lam_V=1e-1):
        super().__init__(M_historical, weighting)
        self.n_iter_cache = n_iter
        if len(M_historical.shape) > 2:
            raise ValueError('Tensor not factorisable.')
        # Initiatialize like netflix prize papers.
        U = np.random.rand(len(M_historical), n_dim+1) / (n_dim+1)
        U[:, -1] = 1.0
        V = np.random.rand(len(M_historical[0])+1, n_dim+1) / (n_dim+1)
        self.U, self.V = U, V
        self.lam_U = lam_U
        self.lam_V = lam_V
        self.n_dim = n_dim

    def fit_predict(self, m_current):
        """See base class."""
        obs, unobs = self.get_obs_ixs(m_current)
        for i in range(self.n_iter_cache):
            observed = np.zeros_like(m_current, dtype=bool)
            observed[obs] = True
            self.update_U(m_current, observed)
            self.update_V(m_current, observed)
        pred = self.U @ self.V[-1, :]
        pred[obs] = m_current[obs]
        return pred

    def update_U(self, m, obs):
        # (1) Update representations of fully observed regions.
        M = np.concatenate([self.M_historical[obs], m[obs].reshape(-1, 1)], 1)
        U = self.U[obs, :-1]
        V, b = self.V[:, :-1], self.V[:, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[obs, :-1] = np.linalg.solve(A, B).T

        # (2) Update representations of other regions.
        M = self.M_historical[~obs]
        U = self.U[~obs, :-1]
        # Only take prior vote representations to update.
        V, b = self.V[:-1, :-1], self.V[:-1, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[~obs, :-1] = np.linalg.solve(A, B).T

    def update_V(self, m, obs):
        # (1) Update all prior vote representations (fully observed ones).
        B = (self.M_historical.T @ self.U).T
        eye = np.eye(self.n_dim+1)
        # Don't regularize bias.
        eye[-1, -1] = 0
        A = self.U.T @ self.U + self.lam_V * eye
        self.V[:-1] = np.linalg.solve(A, B).T

        # (2) Update the new vote representation.
        U = self.U[obs]
        B = (m[obs].reshape(-1, 1).T @ U).T
        A = U.T @ U + self.lam_V * eye
        self.V[-1:] = np.linalg.solve(A, B).T

    def __repr__(self):
        repr_ = 'Matrix Factorization'
        repr_ += ' (dim={}'.format(str(self.n_dim))
        repr_ += ', lam_V={}'.format(str(self.lam_V))
        repr_ += ', lam_U={})'.format(str(self.lam_U))
        return repr_


class SubSVD(Model):
    r"""Describes the base model for the main algorithm of this library.

    This is the basic model as proposed in _Sub-Matrix Factorization for
    Real-Time Vote Prediction KDD'20 by A. Immer and V. Kristof et al.  The
    `M_historical` fully observed matrix is decomposed using the SVD which is
    the optimal solution to non-regularized Matrix Factorization. In this case,
    `M_historical` is `Regions x Votes`. A generalized linear model (GLM) is
    applied to the low-rank regional representations in `self.U`.

    Here, we implement a Gaussian, Bernoulli, and Categorical likelihoods due
    to their relevance in political forecasting.  Let U denote the regional
    features obtained due to the SubSVD.  We minimize the following loss now
    with w_i being `self.weighting[i]` and further assuming
    `self.weighting.sum() == len(self.weighting)`:
    ``l(w) = - sum_{i \\in Observed} w_i \log p(y|U_i^T w + b)
             + l2_reg/ 2 \|w\|_2^2,``
    where U are the regional features and w the parameter of the GLM.
    b is either 0 if `add_bias=False` or otherwise an optimized parameter.

    Attributes:
        n_dim: Number of dimensions in the low-rank representation of
            `M_historical`; determines the dimensionality of latent regional
            features.
        add_bias: Determines whether a bias should be added (default: True).
        l2_reg: L2-regularizer, in line with `lam_U/V` in the
            MatrixFactorization class.
        keep_svals: Determines whether to keep the singular values factored
            into the regional representations in `self.U`, i.e., the features
            for the GLM. If `l2_reg==0/None` this has no effect. Otherwise it
            determines the feature importances (default: True.
    """

    def __init__(self,
                 M_historical,
                 weighting=None,
                 n_dim=10,
                 add_bias=True,
                 l2_reg=1e-5,
                 keep_svals=True):

        if M_historical.ndim > 2:
            raise ValueError('Tensor not factorizable. Use TensorSubSVD.')
        super().__init__(M_historical, weighting)
        U, s, _ = np.linalg.svd(M_historical)
        self.n_dim = n_dim
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias


class GaussianSubSVD(SubSVD):
    """WeightedSubSVD model with Gaussian likelihood in the GLM.

    The Gaussian likelihood has unit variance.

    Attributes:
        See base class.
    """

    def fit_predict(self, m_current):
        obs, _ = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[obs], np.copy(m_current)[obs], self.weighting[obs]
        if self.l2_reg is None or self.l2_reg == 0:
            self.l2_reg = 1 / LARGE_FLOAT
        ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
        ridge.fit(Uo, mo, sample_weight=wo)
        pred = ridge.predict(self.U)
        pred[obs] = m_current[obs]
        return pred

    def __repr__(self):
        repr_ = 'GaussianSubSVD'
        repr_ += ' (dim={}'.format(str(self.n_dim))
        repr_ += ', l2={})'.format(str(self.l2_reg))
        return repr_


class LogisticSubSVD(SubSVD):
    """SubSVD model with Bernoulli likelihood in the GLM.

    Logistic refers to both Bernoulli and categorical likelihoods.
    Categorical is used in the Tensor case. Here, we have binary outcomes.

    Attributes:
        See base class.
    """

    @staticmethod
    def transform_problem(Uo, mo, wo):
        """Transform observations p in [0,1] to a binary classificaiton problem
        by turning p into a weight and having both observation 0 and 1.

        To do so, we repeat the dataset for label y==1 and y==0 and use the
        probabilities as weights. This is equal to cross-entropy regression.
        """

        n = Uo.shape[0]
        wo = np.tile(wo, 2)
        y = np.zeros(2*n)
        y[:n] = 1
        wts = np.tile(mo, 2)
        wts[n:] = 1 - wts[n:]
        wts = wts * wo
        X = np.tile(Uo, (2, 1))
        return X, y, wts

    def fit_predict(self, m_current):
        """See base class."""
        obs, unobs = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[obs], m_current[obs], self.weighting[obs]
        C = LARGE_FLOAT if self.l2_reg == 0 else 1 / self.l2_reg
        logreg = LogisticRegression(C=C, fit_intercept=self.add_bias,
                                    solver='liblinear', tol=1e-6, max_iter=500)
        X, y, wts = self.transform_problem(Uo, mo, wo)
        logreg.fit(X, y, sample_weight=wts)
        pred = logreg.predict_proba(self.U)[:, 1]
        pred[obs] = m_current[obs]
        return pred

    def __repr__(self):
        repr_ = 'LogisticSubSVD'
        repr_ += ' (dim={}'.format(str(self.n_dim))
        repr_ += ', l2={})'.format(str(self.l2_reg))
        return repr_


class TensorSubSVD(Model):
    """Describes the base model for the multiple outcome cases.

    Corresponding base model to SubSVD but applicable to non-binary elections.
    For non-binary outcomes, we assume a tensor of `Regions x Votes x Parties`.
    _Parties_ refers to the amount of options a voter has.

    In comparison to SubSVD, the tensor needs to be collapsed first to apply
    the SVD. This is simply done by forming a `Regions x (Votes * Parties)`
    matrix (reshaping). The GLM that is applied works equivalently as before:
    Using the regional feature vector, a linear model predicts the
    _Parties_-dimensional outcome vector.

    In the subclasses, we have either a multivariate Gaussian likelihood
    or a categorical likelihood.

    Attributes:
        n_dim:
            Number of retained dimensions in the low-rank representation of
            `M_historical`; determines the dimensionality of latent regional
            features.
        add_bias:
            Determines whether a bias should be added (default: True).
        l2_reg:
            L2-regularizer, in line with `lam_U/V` in the MatrixFactorization
            class.
        keep_svals: Determines whether to keep the singular values factored
            into the regional representations in `self.U` which are the
            features for the GLM. If `l2_reg==0/None` this has no effect.
            Otherwise it determines the feature importances.
    """

    def __init__(self, M_historical, weighting, n_dim=10, add_bias=True,
                 l2_reg=1e-5, keep_svals=True):
        if len(M_historical.shape) < 3:
            raise ValueError('Requires Tensor')
        super().__init__(M_historical, weighting)
        M_historical = M_historical.reshape(M_historical.shape[0], -1)
        U, s, _ = np.linalg.svd(M_historical)
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias
        self.n_dim = n_dim


class GaussianTensorSubSVD(TensorSubSVD):
    """TensorSubSVD with multivariate Gaussian likelihood for the GLM."""

    def fit_predict(self, m_current):
        obs, _ = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[obs], m_current[obs], self.weighting[obs]
        if self.l2_reg == 0:
            self.l2_reg = 1 / LARGE_FLOAT
        ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
        ridge.fit(Uo, mo, sample_weight=wo)
        pred = ridge.predict(self.U)
        pred[obs] = m_current[obs]
        return pred

    def __repr__(self):
        repr_ = 'GaussianTensorSubSVD'
        repr_ += ' (dim={}'.format(str(self.n_dim))
        repr_ += ', l2={})'.format(str(self.l2_reg))
        return repr_


class LogisticTensorSubSVD(TensorSubSVD):
    """TensorSubSVD with Categorical likelihood with _Parties_ categories.
    All parameters are as in TensorSubSVD but we additionally initialize
    the logistic model (categorical GLM) for warmstarts.
    """

    def __init__(self, M_historical, weighting, n_dim=10, add_bias=True,
                 l2_reg=1e-5, keep_svals=True):
        super().__init__(M_historical, weighting,
                         n_dim, add_bias, l2_reg, keep_svals)
        C = LARGE_FLOAT if self.l2_reg == 0 else 1 / self.l2_reg
        self.model = LogisticRegression(C=C, fit_intercept=add_bias, tol=1e-6,
                                        solver='newton-cg', max_iter=5000,
                                        multi_class='multinomial', n_jobs=4,
                                        warm_start=True)

    @staticmethod
    def transform_problem(Uo, mo, wo):
        """Transform non-categorical floating outcomes to categorical outcomes
        and corresponding weights (see LogisticSubSVD.transform_problem).

        The _Parties_-dimensional rows of `mo` need to sum to 1 but each entry
        can be a float in `[0,1]`.
        Then, we make _Parties_ observations out of this that are categorical
        and each is weighted by the probability indicated as in `mo`.
        """
        n, k = mo.shape
        wo = np.tile(wo, k)
        # Classes 0*n, 1*n, ..., k*n.
        y = np.arange(k).repeat(n)
        # Weights are probabilities of respective class.
        wts = mo.reshape(-1, order='F')
        # Repeat data.
        X = np.tile(Uo, (k, 1))
        wts = wts * wo
        return X, y, wts

    def fit_predict(self, m_current):
        obs, unobs = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[obs], m_current[obs], self.weighting[obs]
        X, y, wts = self.transform_problem(Uo, mo, wo)
        self.model.fit(X, y, sample_weight=wts)
        pred = self.model.predict_proba(self.U)
        pred[obs] = m_current[obs]
        return pred

    def __repr__(self):
        repr_ = 'LogisticTensorSubSVD'
        repr_ += ' (dim={}'.format(str(self.n_dim))
        repr_ += ', l2={})'.format(str(self.l2_reg))
        return repr_
