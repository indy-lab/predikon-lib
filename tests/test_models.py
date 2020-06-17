import pytest
import numpy as np

from predikon import Model, Averaging, WeightedAveraging


def get_M_w_vec():
    M = np.random.randn(3, 4)
    w = np.abs(np.random.randn(3))
    return M, w


def get_M_w_mat():
    M = np.random.randn(3, 4, 2)
    w = np.abs(np.random.randn(3, 2))
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


