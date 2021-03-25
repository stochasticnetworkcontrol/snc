import numpy as np

from src.snc.agents.hedgehog import safety_stocks
import pytest


def test_map_workload_vectors_to_physical_resources_no_valid_allocation():
    load_wl = np.array([0.6, 0.7, 0.8, 0.9])[:, None]
    sigma_2_wl = np.array([1, 2, 3, 4])[:, None]
    nu = np.zeros((4, 5))
    with pytest.raises(AssertionError):
        _, _ = safety_stocks.map_workload_to_physical_resources_with_conservative_max_heuristic(
            nu, load_wl, sigma_2_wl)


def test_map_workload_vectors_to_physical_resources_less_physical_than_workload_resources():
    load_wl = np.array([0.6, 0.7, 0.8, 0.9])[:, None]
    sigma_2_wl = np.array([1, 2, 3, 4])[:, None]
    nu = np.array([[1, 0, 0, 0],
                   [0, 0.5, 0.5, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0.3, 0.7]])
    load_rs, sigma_2_rs = \
        safety_stocks.map_workload_to_physical_resources_with_conservative_max_heuristic(
            nu, load_wl, sigma_2_wl)
    assert np.all(load_rs == np.array([0.6, 0.8, 0.9, 0.9])[:, None])
    assert np.all(sigma_2_rs == np.array([1, 3, 4, 4])[:, None])


def test_map_workload_vectors_to_physical_resources_more_physical_than_workload_resources():
    load_wl = np.array([0.6, 0.7, 0.8])[:, None]
    sigma_2_wl = np.array([1, 2, 3])[:, None]
    nu = np.array([[1, 0, 0, 0],
                   [0, 0.5, 0.5, 0],
                   [0, 0, 0.3, 0.7]])
    load_rs, sigma_2_rs = \
        safety_stocks.map_workload_to_physical_resources_with_conservative_max_heuristic(
            nu, load_wl, sigma_2_wl)
    assert np.all(load_rs == np.array([0.6, 0.7, 0.8, 0.8])[:, None])
    assert np.all(sigma_2_rs == np.array([1, 2, 3, 3])[:, None])


def test_obtain_safety_stock_per_resource_zero_theta():
    theta = 0  # We have to avoid dividing by zero
    load_s = 0
    sigma_2 = 0
    state = np.zeros((3, 1))
    with pytest.raises(AssertionError):
        _ = safety_stocks.obtain_safety_stock_per_resource(theta, load_s, sigma_2, state)


def test_obtain_safety_stock_per_resource_too_large_load():
    theta = 1
    load_s = 1  # Load is too large
    sigma_2 = 1
    state = np.ones((3, 1))
    with pytest.raises(AssertionError):
        _ = safety_stocks.obtain_safety_stock_per_resource(theta, load_s, sigma_2, state)


def test_obtain_safety_stock_per_resource_some_values_1():
    theta = 1
    load_s = 0.9
    sigma_2 = 0
    state = np.array([0.1, 0.3, 2])[:, None]
    tau = safety_stocks.obtain_safety_stock_per_resource(theta, load_s, sigma_2, state)
    assert tau == 0


def test_obtain_safety_stock_per_resource_some_values_2():
    theta = 1
    load_s = 0.9
    sigma_2 = 1.1
    state = np.array([0.1, 0.3, 2])[:, None]
    tau = safety_stocks.obtain_safety_stock_per_resource(theta, load_s, sigma_2, state)
    np.testing.assert_almost_equal(tau, 3.310543013, decimal=6)


def test_obtain_safety_stock_per_resource_some_values_3():
    theta = 3.5
    load_s = 0.8
    sigma_2 = 3.9
    state = np.array([4, 5, 6])[:, None]
    tau = safety_stocks.obtain_safety_stock_per_resource(theta, load_s, sigma_2, state)
    np.testing.assert_almost_equal(tau, 15.5315877, decimal=6)


def test_obtain_safety_stock_per_resource_for_getting_tau():
    theta = np.array([2, 3, 6])
    load = np.array([0.9, 2.8 / 3, 5.8 / 6])[:, None]
    sigma_2 = (np.exp(1) - 1) * 0.2 / 5
    state = np.array([0.5, 1, 1.5, 2])[:, None]
    tau = np.zeros((3, 1))
    for s in range(3):
        tau[s] = safety_stocks.obtain_safety_stock_per_resource(theta[s], load[s], sigma_2, state)

    np.testing.assert_almost_equal(tau, np.array([2, 3, 6])[:, None], decimal=6)


def test_obtain_safety_stock_vector_some_values_1():
    theta = np.array([1, 1])[:, None]
    load = np.array([0.9, 0.9])[:, None]
    sigma_2 = np.array([0, 1.1])[:, None]
    state = np.array([0.1, 0.3, 2])[:, None]
    tau = safety_stocks.obtain_safety_stock_vector(theta, load, sigma_2, state)
    np.testing.assert_allclose(tau, np.array([3.218876, 3.310543013])[:, None], rtol=1e-4)


def test_obtain_safety_stock_vector_some_values_2():
    theta = np.array([2, 3, 6])
    load = np.array([0.9, 2.8 / 3, 5.8 / 6])[:, None]
    sigma_2 = (np.exp(4) - 1) * 0.2 / 5 * np.ones_like(load)
    state = np.array([0.5, 1, 1.5, 2])[:, None]
    tau = safety_stocks.obtain_safety_stock_vector(theta, load, sigma_2, state)
    np.testing.assert_allclose(tau, np.array([8, 12, 24])[:, None], rtol=1e-4)


def test_obtain_gradient_penalty_per_condition_resource_wrong_dimensionality():
    kappa_s = 0
    tau_s = 0
    boundary_condition_sj = np.ones((4, 1))
    state = np.ones((2, 1))
    with pytest.raises(AssertionError):
        _ = safety_stocks.obtain_penalty_gradient_per_condition_resource(kappa_s, tau_s,
                                                                         boundary_condition_sj,
                                                                         state)


def test_obtain_gradient_penalty_per_condition_resource_zero_threshold():
    kappa_s = 0
    tau_s = 0
    boundary_condition_sj = np.array([1, 1, 0, 0])
    state = np.array([1, 1, 1, 1])[:, None]
    psi_sj = safety_stocks.obtain_penalty_gradient_per_condition_resource(kappa_s, tau_s,
                                                                          boundary_condition_sj,
                                                                          state)
    # b = state \cdot boundary_condition_sj
    # psi = - kappa_s * b * boundary_condition_sj
    assert np.all(psi_sj == np.zeros((4, 1)))


def test_obtain_gradient_penalty_per_condition_resource_zero_buffer():
    kappa_s = 1
    tau_s = 1
    boundary_condition_sj = np.array([1, 1, 1, 1])
    state = np.zeros((4, 1))
    psi_sj = safety_stocks.obtain_penalty_gradient_per_condition_resource(kappa_s, tau_s,
                                                                          boundary_condition_sj,
                                                                          state)
    # b = state \cdot boundary_condition_sj
    # psi = - kappa_s * b * boundary_condition_sj
    assert np.all(psi_sj == - np.ones((4, 1)))  # - (1 - 0) * [[1, 1, 1, 1]]


def test_obtain_gradient_penalty_per_condition_resource_enough_customers():
    kappa_s = 1
    tau_s = 2
    boundary_condition_sj = np.array([1, 1, 0, 0])
    state = np.array([1, 1, 0, 0])[:, None]
    psi_sj = safety_stocks.obtain_penalty_gradient_per_condition_resource(kappa_s, tau_s,
                                                                          boundary_condition_sj,
                                                                          state)
    # b = state \cdot boundary_condition_sj
    # psi = - kappa_s * b * boundary_condition_sj
    assert np.all(psi_sj == np.zeros((4, 1)))  # - (2 - 2) * [[1, 1, 0, 0]]


def test_obtain_gradient_penalty_per_condition_resource_not_enough_customers():
    kappa_s = 1
    tau_s = 3
    boundary_condition_sj = np.array([1, 1, 0, 0])
    state = np.array([1, 1, 0, 0])[:, None]
    psi_sj = safety_stocks.obtain_penalty_gradient_per_condition_resource(kappa_s, tau_s,
                                                                          boundary_condition_sj,
                                                                          state)
    # b = state \cdot boundary_condition_sj
    # psi = - kappa_s * b * boundary_condition_sj
    assert np.all(psi_sj == - boundary_condition_sj[:, None])  # - (3 - 2) = 1
