import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression


class Model:
    """
    Base class for online predictive models for regional results.

    Generally, subclasses of this model can be used in the following way
    ```
    import numpy as np
    R, V = 100, 10
    # M is Region x Votes or Region x Votes x Parties
    M_historical = np.random.randn(R, V)
    m_current = np.random.randn(R)
    # TODO: set some nans etc.
    # weighting is R dimensional associated with each region
    weightings = np.abs(np.random.randn(R))
    m = Model(M_historical, weighting)
    pred = m.fit_predict(m_current)
    # pred has filled nans with predictions.
    ```
    """

    def __init__(self, M_historical, weighting):
        """
        Parameters
        ----------
        M_historical : np.array
            RxV or RxVxP dimensional array with outcomes in [0,1].
            R regions, V votes/elections, P parties (if non-binary outcome)
        weighting : np.array
            R dimensional array with weights associated to each region
            if `weighting=None` the algorithm is unweighted
            (equivalent to np.ones(R))
        """
        self.M_historical = M_historical
        # make weights sum to R
        self.weighting = weighting / np.sum(weighting) * M_historical.shape[0]

    def fit_predict(self, m_current):
        """
        Fit the model to current observations and predict on the missing
        entries of `m_current` as indicated by np.nans.
        Parameters
        ----------
        m_current : np.array
            R or RxP dimensional array with outcomes in [0,1] and unobserved
            entries which are given as np.nan
        """
        raise NotImplementedError

    def get_obs_ixs(self, m_current):
        """
        Turn the array `m_current` into two index-lists of observed and
        unobserved regions by detecting unobserved nan-values.
        Parameters
        ----------
        m_current : np.array
            R or RxP dimensional array with outcomes in [0,1] and unobserved
            entries which are given as np.nan
        """
        if m_current.ndim == 1:
            assert len(m_current) == self.M_historical.shape[0]
            # R - vector
            unobserved = np.isnan(m_current)
        elif m_current.ndim == 2:
            R, P = m_current.shape
            assert (R == self.M_historical.shape[0] and
                    P == self.M_historical.shape[2])
            # R x P dim
            unobserved = np.any(np.isnan(m_current), axis=-1)

        m_obs_ixs, m_unobs_ixs = np.where(~unobserved), np.where(unobserved)
        return m_obs_ixs, m_unobs_ixs

    def __repr__(self):
        return 'Predikon BaseModel'


class Averaging(Model):

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        pred = m_current[m_obs_ixs].mean(axis=0)
        pred_col = np.ones_like(m_current) * pred
        pred_col[m_obs_ixs] = m_current[m_obs_ixs]
        return pred_col

    def __repr__(self):
        return 'Averaging'


class WeightedAveraging(Model):

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        wts = self.weighting[m_obs_ixs]
        N = wts.sum()
        if len(m_current.shape) > 1:
            raise ValueError('Multi-dim not available.')
        pred = (m_current[m_obs_ixs] * wts).sum() / N
        pred_col = np.ones_like(m_current) * pred
        pred_col[m_obs_ixs] = m_current[m_obs_ixs]
        return pred_col

    def __repr__(self):
        return 'Weighted Averaging'


class MatrixFactorisation(Model):

    def __init__(self, M_historical, weighting, n_iter=20, n_dim=25,
                 lam_U=1e-1, lam_V=1e-1):
        super().__init__(M_historical, weighting)
        self.n_iter_cache = n_iter
        if len(M_historical.shape) > 2:
            raise ValueError('Tensor not factorisable.')
        # initiatialize like netflix prize papers
        U = np.random.rand(len(M_historical), n_dim+1) / (n_dim+1)
        U[:, -1] = 1.0
        V = np.random.rand(len(M_historical[0])+1, n_dim+1) / (n_dim+1)
        self.U, self.V = U, V
        self.lam_U = lam_U
        self.lam_V = lam_V
        self.n_dim = n_dim

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        for i in range(self.n_iter_cache):
            observed = np.zeros_like(m_current, dtype=bool)
            observed[m_obs_ixs] = True
            self.update_U(m_current, observed)
            self.update_V(m_current, observed)
        return self.U @ self.V[-1, :]

    def update_U(self, m, observed):
        # (1) update representations of fully observed regions
        M = np.concatenate([self.M_historical[observed], m[observed].reshape(-1, 1)], 1)
        U = self.U[observed, :-1]
        V, b = self.V[:, :-1], self.V[:, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[observed, :-1] = np.linalg.solve(A, B).T

        # (2) update representations of other regions
        M = self.M_historical[~observed]
        U = self.U[~observed, :-1]
        # only take prior vote representations to update
        V, b = self.V[:-1, :-1], self.V[:-1, -1]
        ones = np.ones(len(U))
        B = ((M - np.outer(ones, b)) @ V).T
        A = V.T @ V + self.lam_U * np.eye(self.n_dim)
        self.U[~observed, :-1] = np.linalg.solve(A, B).T

    def update_V(self, m, observed):
        # (1) update all prior vote representations (fully observed ones)
        V = self.V[:-1]
        B = (self.M_historical.T @ self.U).T
        I = np.eye(self.n_dim+1)
        # don't regularize bias
        I[-1, -1] = 0
        A = self.U.T @ self.U + self.lam_V * I
        self.V[:-1] = np.linalg.solve(A, B).T

        # (2) update the new vote representation
        U = self.U[observed]
        V = self.V[-1:]
        B = (m[observed].reshape(-1, 1).T @ U).T
        A = U.T @ U + self.lam_V * I
        self.V[-1:] = np.linalg.solve(A, B).T

    def __repr__(self):
        repr_ =  'Matrix Factorization'
        repr_ += (' (dim=' + str(self.n_dim) + ',lam_V=' + str(self.lam_V) +
                 ',lam_U=' + str(self.lam_U) + ')')
        return repr_


class SubSVD(Model):

    def __init__(self, M_historical, weighting,
                 n_dim=10, add_bias=True, l2_reg=1e-5, keep_svals=True):
        if len(M_historical.shape) > 2:
            raise ValueError('Tensor not factorisable.')
        super().__init__(M_historical, weighting)
        U, s, _ = np.linalg.svd(M_historical)
        self.n_dim = n_dim
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo = self.U[m_obs_ixs], m_current[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo)
            return ridge.predict(self.U)
        else:
            # TODO: add bias for non-regularized model
            x, _, _, _ = np.linalg.lstsq(Uo, mo, 1e-9)
            return self.U @ x

    def __repr__(self):
        return ('SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class WeightedSubSVD(SubSVD):

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[m_obs_ixs], m_current[m_obs_ixs], self.weighting[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo, sample_weight=wo)
            return ridge.predict(self.U)
        else:
            # TODO: add bias for non-regularized model
            wo_sqrt = np.sqrt(wo)
            x, _, _, _ = np.linalg.lstsq(
                Uo * wo_sqrt.reshape(-1, 1), mo * wo_sqrt, 1e-9)
            return self.U @ x

    def __repr__(self):
        return ('Weighted SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class LogisticSubSVD(SubSVD):

    @staticmethod
    def transform_problem(Uo, mo, wo=None):
        # Repeat dataset for label y==1 and y==0 and use the probabilities
        # as weights. This is equal to cross-entropy logistic regression.
        n = Uo.shape[0]
        wo = np.ones(n) if wo is None else wo / wo.sum() * n
        wo = np.tile(wo, 2)
        y = np.zeros(2*n)
        y[:n] = 1
        wts = np.tile(mo, 2)
        wts[n:] = 1 - wts[n:]
        wts = wts * wo
        X = np.tile(Uo, (2, 1))
        return X, y, wts

    def fit_predict(self, m):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo = self.U[m_obs_ixs], m_current[m_obs_ixs]
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        logreg = LogisticRegression(C=C, fit_intercept=self.add_bias,
                                    solver='liblinear', tol=1e-6, max_iter=500)
        X, y, wts = self.transform_problem(Uo, mo, None)
        logreg.fit(X, y, sample_weight=wts)
        return logreg.predict_proba(self.U)[:, 1]

    def __repr__(self):
        return ('Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class WeightedLogisticSubSVD(LogisticSubSVD):

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo, wo = self.U[m_obs_ixs], m_current[m_obs_ixs], self.weighting[m_obs_ixs]
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        logreg = LogisticRegression(C=C, fit_intercept=self.add_bias,
                                    solver='liblinear', tol=1e-6, max_iter=500)
        X, y, wts = self.transform_problem(Uo, mo, wo)
        logreg.fit(X, y, sample_weight=wts)
        return logreg.predict_proba(self.U)[:, 1]

    def __repr__(self):
        return ('Weighted Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class TensorSubSVD(Model):
    """Folds the m by v by party tensor into m by (v*party) and then
    factorize"""

    def __init__(self, M_historical, weighting, n_dim=10, add_bias=True, l2_reg=1e-5,
                 keep_svals=True):
        if len(M_historical.shape) < 3:
            raise ValueError('Requires Tensor')
        super().__init__(M_historical, weighting)
        M_historical = M_historical.reshape(M_historical.shape[0], -1)
        U, s, _ = np.linalg.svd(M_historical)
        self.U = U[:, :n_dim] * s[None, :n_dim] if keep_svals else U[:, :n_dim]
        self.l2_reg = l2_reg
        self.add_bias = add_bias
        self.n_dim = n_dim

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo = self.U[m_obs_ixs], m_current[m_obs_ixs]
        if self.l2_reg is not None and self.l2_reg != 0:
            ridge = Ridge(alpha=self.l2_reg, fit_intercept=self.add_bias)
            ridge.fit(Uo, mo)
            return ridge.predict(self.U)
        else:
            # TODO: add bias here?
            x, _, _, _ = np.linalg.lstsq(Uo, mo, 1e-9)
            return self.U @ x

    def __repr__(self):
        return ('SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')


class LogisticTensorSubSVD(TensorSubSVD):

    def __init__(self, M_historical, weighting, n_dim=10, add_bias=True, l2_reg=1e-5,
                 keep_svals=True):
        super().__init__(M_historical, weighting, n_dim, add_bias, l2_reg, keep_svals)
        C = 0 if self.l2_reg == 0 else 1 / self.l2_reg
        self.model = LogisticRegression(C=C, fit_intercept=add_bias, tol=1e-6,
                                        solver='newton-cg', max_iter=5000,
                                        multi_class='multinomial', n_jobs=4,
                                        warm_start=True)

    @staticmethod
    def transform_problem(Uo, mo):
        n, k = mo.shape
        # classes 0*n, 1*n, ..., k*n
        y = np.arange(k).repeat(n)
        # weights are probabilities of respective class
        wts = mo.reshape(-1, order='F')
        # repeat data
        X = np.tile(Uo, (k, 1))
        return X, y, wts

    def fit_predict(self, m_current):
        m_obs_ixs, m_unobs_ixs = self.get_obs_ixs(m_current)
        Uo, mo = self.U[m_obs_ixs], m_current[m_obs_ixs]
        X, y, wts = self.transform_problem(Uo, mo)
        self.model.fit(X, y, sample_weight=wts)
        return self.model.predict_proba(self.U)

    def __repr__(self):
        return ('Logistic SubSVD' + ' (dim=' + str(self.n_dim) + ',l2=' +
                str(self.l2_reg) + ')')

