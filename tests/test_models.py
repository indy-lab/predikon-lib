import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from predikon import (Model, WeightedAveraging, MatrixFactorisation,
                      GaussianSubSVD, LogisticSubSVD, GaussianTensorSubSVD,
                      LogisticTensorSubSVD)


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
    # Models = [GaussianSubSVD, LogisticSubSVD]
    M, w = get_M_w_vec()
    m = np.array([0.0, 0.3, np.nan])
    w = np.array([2, 7, 2])

    # Gaussian
    model = GaussianSubSVD(M, w, n_dim=1, l2_reg=1e-5)
    pred = model.fit_predict(m)
    # assert not (np.isnan(pred[-1]))
    # assert np.allclose(pred[:2], m[:2])
    assert_almost_equal(pred[:2], m[:2])

    # Bernoulli
    model = LogisticSubSVD(M, w, n_dim=1, l2_reg=1e-5)
    pred = model.fit_predict(m)
    # assert not (np.isnan(pred[-1]))
    # assert np.allclose(pred[:2], m[:2])
    assert_almost_equal(pred[:2], m[:2])

    # for MODEL in Models:
    #     model = MODEL(M, w, n_dim=1, l2_reg=1e-5)
    #     pred = model.fit_predict(m)
    #     assert not (np.isnan(pred[-1]))
    #     assert np.allclose(pred[:2], m[:2])


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
    error_init = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    pred = model.fit_predict(m)
    error_trained = susq(Ms - model.U @ model.V.T) + susq(model.U) + susq(model.V)
    assert error_trained < error_init


"""Exception Tests"""


def test_unallowed_weighting_length():
    with pytest.raises(ValueError, match=r".*dimension.*"):
        M, w = get_M_w_vec()
        w = w[:-1]
        model = Model(M, w)


def test_unallowed_weighting_shape():
    with pytest.raises(ValueError, match=r".*vector.*"):
        M, w = get_M_w_vec()
        w = np.ones((len(w), len(w)))
        model = Model(M, w)
