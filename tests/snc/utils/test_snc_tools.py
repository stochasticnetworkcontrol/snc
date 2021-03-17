import numpy as np
import snc.utils.snc_tools as snc_tools


def test_is_binary_ones():
    a = np.ones((2, 5))
    assert snc_tools.is_binary(a)


def test_is_binary_zeros():
    a = np.zeros((2, 5))
    assert snc_tools.is_binary(a)


def test_is_binary():
    a = np.zeros((2, 5))
    a[1, 1] = 1
    assert snc_tools.is_binary(a)


def test_is_binary_not_binary():
    a = np.ones((2, 5))
    a[1, 1] = 0.1
    assert not snc_tools.is_binary(a)


def test_is_approx_binary_ones():
    a = np.ones((2, 5)) + 1e-7
    assert snc_tools.is_approx_binary(a)


def test_is_approx_binary_ones_negative_eps():
    a = np.ones((2, 5)) - 1e-7
    assert snc_tools.is_approx_binary(a)


def test_is_approx_binary_zeros():
    a = np.zeros((2, 5)) + 1e-7
    assert snc_tools.is_approx_binary(a)


def test_is_approx_binary_zeros_negative_eps():
    a = np.zeros((2, 5)) - 1e-7
    assert snc_tools.is_approx_binary(a)


def test_is_approx_binary_ones_too_high_eps():
    a = np.zeros((2, 5)) + 1e-3
    assert not snc_tools.is_approx_binary(a)


def test_is_approx_binary_ones_custom_tol_positive():
    eps = 1e-3
    a = np.ones((2, 5)) + eps
    assert snc_tools.is_approx_binary(a, eps)


def test_is_approx_binary_ones_custom_tol_negative():
    eps = 1e-3
    a = np.ones((2, 5)) - eps
    assert snc_tools.is_approx_binary(a, eps)


def test_is_approx_binary_zeros_custom_tol_positive():
    eps = 1e-3
    a = np.zeros((2, 5)) + eps
    assert snc_tools.is_approx_binary(a, eps)


def test_is_approx_binary_zeros_custom_tol_negative():
    eps = 1e-3
    a = np.zeros((2, 5)) - eps
    assert snc_tools.is_approx_binary(a, eps)


def test_is_binary_negative():
    a = np.zeros((2, 5))
    a[1, :] = - np.ones((5, ))
    assert snc_tools.is_binary_negative(a)


def test_is_binary_negative_fault():
    a = np.zeros((2, 5))
    a[1, 1] = 1
    assert not snc_tools.is_binary_negative(a)

