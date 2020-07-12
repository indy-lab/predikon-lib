import pytest
import numpy as np

from predikon import (Model, Averaging, WeightedAveraging, MatrixFactorisation,
                      GaussianSubSVD, WeightedGaussianSubSVD, LogisticSubSVD,
                      WeightedLogisticSubSVD, GaussianTensorSubSVD,
                      LogisticTensorSubSVD)


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


def test_averaging():
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    model = Averaging(M, w)
    pred = model.fit_predict(m)
    assert pred[-1] == 0.15


def test_averaging_mat():
    M, w = get_M_w_mat()
    m = np.array([0.0, 0.3, np.nan])
    m = np.stack([m, m]).T
    model = Averaging(M, w)
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


def test_prediction_not_nan_vec():
    Models = [MatrixFactorisation, GaussianSubSVD, LogisticSubSVD,
              WeightedGaussianSubSVD, WeightedLogisticSubSVD]
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])
    for MODEL in Models:
        model = MODEL(M, w, n_dim=1)
        pred = model.fit_predict(m)
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


def test_mf_converges():
    np.random.seed(235)
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, 0.7])
    w = np.array([2, 7, 2])
    model = MatrixFactorisation(M, w, n_dim=1)
    Ms = np.concatenate([M, m.reshape(-1, 1)], 1)
    error_init = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    pred = model.fit_predict(m)
    error_trained = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    assert error_trained < error_init
