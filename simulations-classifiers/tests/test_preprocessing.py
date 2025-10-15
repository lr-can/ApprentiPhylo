import numpy as np

from classifiers.data import preprocessing_fn


def test_site_composition():
    align = np.array([[0, 2, 2], [1, 2, 4]])
    res = preprocessing_fn.site_composition(align, n_values=5)
    assert np.array_equal(res, np.array([[0.5, 0.5, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0.5, 0, 0.5]]))
    res = preprocessing_fn.site_composition(align, n_values=6)
    assert np.array_equal(res, np.array([[0.5, 0.5, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0.5, 0, 0.5, 0]]))


def test_msa_composition():
    align = np.array([[0, 2, 2, 2], [1, 2, 4, 0]])
    res = preprocessing_fn.msa_composition(align, n_values=5)
    assert np.array_equal(res, np.array([0.25, 0.125, 0.5, 0, 0.125]))
    res = preprocessing_fn.msa_composition(align, n_values=6)
    assert np.array_equal(res, np.array([0.25, 0.125, 0.5, 0, 0.125, 0]))
