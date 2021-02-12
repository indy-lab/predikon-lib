import pytest
import numpy as np

from predikon import (Model, WeightedAveraging, MatrixFactorisation,
                      GaussianSubSVD, BayesianSubSVD, LogisticSubSVD,
                      GaussianTensorSubSVD, LogisticTensorSubSVD)


"""Setup methods"""


def susq(A):
    # sum of squares
    return np.sum(np.square(A))


def get_M_w_vec():
    M = np.random.randn(3, 4)
    w = np.abs(np.random.randn(3))
    return M, w


def get_M_w_mat():
    # Region x Vote x Party
    M = np.random.randn(3, 4, 2)
    w = np.abs(np.random.randn(3))
    return M, w


"""Programmatical Tests"""


def test_observed_indexes_vec():
    M, w = get_M_w_vec()
    m = np.array([0, np.nan, 0])
    obs_ixs = np.array([0, 2])
    unobs_ixs = np.array([1])
    model = Model(M, w)
    obs, unobs = model.get_obs_ixs(m)
    assert np.all(np.equal(obs, obs_ixs))
    assert np.all(np.equal(unobs, unobs_ixs))


def test_observed_indexes_mat():
    M, w = get_M_w_mat()
    m = np.array([0, np.nan, 0])
    m = np.stack([m, m]).T
    obs_ixs = np.array([0, 2])
    unobs_ixs = np.array([1])
    model = Model(M, w)
    obs, unobs = model.get_obs_ixs(m)
    assert np.all(np.equal(obs, obs_ixs))
    assert np.all(np.equal(unobs, unobs_ixs))


def test_prediction_not_nan_vec():
    Models = [MatrixFactorisation, GaussianSubSVD, LogisticSubSVD]
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])
    for MODEL in Models:
        model = MODEL(M, w, n_dim=1)
        pred = model.fit_predict(m)
        assert not (np.isnan(pred[-1]))
    # separately test for Bayesian
    model = BayesianSubSVD(M, w, n_dim=1)
    pred, pred_std = model.fit_predict(m)
    assert not (np.isnan(pred[-1]))


def test_prediction_not_nan_mat():
    Models = [GaussianTensorSubSVD, LogisticTensorSubSVD]
    M, w = get_M_w_mat()
    m = np.array([0.0, 0.3, np.nan])
    m = np.stack([m, 1-m]).T
    for MODEL in Models:
        model = MODEL(M, w)
        pred = model.fit_predict(m)
        assert not any(np.isnan(pred[-1]))


def test_prediction_not_nan_vec_unreg():
    Models = [GaussianSubSVD, LogisticSubSVD]
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])
    for MODEL in Models:
        model = MODEL(M, w, n_dim=1, l2_reg=0)
        pred = model.fit_predict(m)
        assert not (np.isnan(pred[-1]))


def test_prediction_not_nan_mat_unreg():
    Models = [GaussianTensorSubSVD, LogisticTensorSubSVD]
    M, w = get_M_w_mat()
    m = np.array([0.0, 0.3, np.nan])
    m = np.stack([m, 1-m]).T
    for MODEL in Models:
        model = MODEL(M, w)
        pred = model.fit_predict(m)
        assert not any(np.isnan(pred[-1]))


def test_prediction_fill_nan_only():
    models = [GaussianSubSVD, LogisticSubSVD]
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])

    for model in models:
        model = model(M, w, n_dim=1, l2_reg=1e-5)
        pred = model.fit_predict(m)
        assert not (np.isnan(pred[-1]))
        assert np.allclose(pred[:2], m[:2])

    # Bayesian model with std
    model = BayesianSubSVD(M, w, n_dim=1)
    pred, pred_std = model.fit_predict(m)
    assert not (np.isnan(pred[-1]))
    assert np.allclose(pred[:2], m[:2])
    # observed entries have no uncertainty!
    assert np.allclose(pred_std[:2], np.zeros_like(m[:2]))


"""Methodological Tests"""


def test_averaging():
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    model = WeightedAveraging(M, weighting=None)
    pred = model.fit_predict(m)
    assert pred[-1] == 0.15


def test_averaging_mat():
    M, w = get_M_w_mat()
    m = np.array([0.0, 0.3, np.nan])
    m = np.stack([m, m]).T
    model = WeightedAveraging(M, weighting=None)
    pred = model.fit_predict(m)
    assert np.all(pred[-1] == np.array([0.15, 0.15]))


def test_weighted_averaging():
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])
    res = 0.3 * 7 / 9
    model = WeightedAveraging(M, w)
    pred = model.fit_predict(m)
    assert pred[-1] == res


def test_weighted_averaging_mat():
    M, w = get_M_w_mat()
    m = np.array([0.0, 0.3, np.nan])
    m = np.stack([m, 1-m]).T
    w = np.array([2, 7, 2])
    res = np.array([0.3 * 7 / 9, 2 / 9 + 0.7 * 7/9])
    model = WeightedAveraging(M, w)
    pred = model.fit_predict(m)
    # should predict regions x parties
    assert pred.shape == (M.shape[0], M.shape[2])
    assert np.all(pred[-1] == res)


def test_mf_converges():
    np.random.seed(235)
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, 0.7])
    w = np.array([2, 7, 2])
    model = MatrixFactorisation(M, w, n_dim=1)
    Ms = np.concatenate([M, m.reshape(-1, 1)], 1)
    err_init = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    _ = model.fit_predict(m)
    err_train = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    assert err_train < err_init


"""Representation Tests"""


def test_repr_vec():
    # Get data.
    M, w = get_M_w_vec()
    # Set hyperparameters.
    d, l2 = 10, 0.1
    # Define expected representations.
    wa_repr = 'Weighted Averaging'
    mf_repr = 'Matrix Factorization (dim=10, lam_V=0.1, lam_U=0.1)'
    gaus_repr = 'GaussianSubSVD (dim=10, l2=0.1)'
    bern_repr = 'LogisticSubSVD (dim=10, l2=0.1)'
    bays_repr = 'BayesianSubSVD (dim=10)'
    # Test representations.
    MF = MatrixFactorisation
    repr2model = {
        wa_repr: WeightedAveraging(M),
        mf_repr: MF(M, w, n_dim=d, lam_V=l2, lam_U=l2),
        gaus_repr: GaussianSubSVD(M, w, n_dim=d, l2_reg=l2),
        bern_repr: LogisticSubSVD(M, w, n_dim=d, l2_reg=l2),
        bays_repr: BayesianSubSVD(M, w, n_dim=d)
    }
    for repr_, model in repr2model.items():
        assert repr_ == model.__repr__()


def test_repr_mat():
    # Get data.
    M, w = get_M_w_mat()
    # Set hyperparameters.
    d, l2 = 10, 0.1
    # Define expected representations.
    gaus_repr = 'GaussianTensorSubSVD (dim=10, l2=0.1)'
    bern_repr = 'LogisticTensorSubSVD (dim=10, l2=0.1)'
    # Test representations.
    repr2model = {
        gaus_repr: GaussianTensorSubSVD(M, w, n_dim=d, l2_reg=l2),
        bern_repr: LogisticTensorSubSVD(M, w, n_dim=d, l2_reg=l2)
    }
    for repr_, model in repr2model.items():
        assert repr_ == model.__repr__()


"""Exception Tests"""


def test_unallowed_weighting_length():
    with pytest.raises(ValueError, match=r".*dimension.*"):
        M, w = get_M_w_vec()
        w = w[:-1]
        _ = Model(M, w)


def test_unallowed_weighting_shape():
    with pytest.raises(ValueError, match=r".*dimension.*"):
        M, w = get_M_w_vec()
        w = np.ones((len(w), len(w)))
        _ = Model(M, w)
