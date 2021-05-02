import numpy as np
from typing import List, Tuple
import snc.utils.snc_types as types
import math
from collections import defaultdict


def map_workload_to_physical_resources_with_conservative_max_heuristic(
        nu: types.NuMatrix, load_wl: types.WorkloadSpace, sigma_2_wl: types.WorkloadSpace) \
        -> Tuple[types.ResourceSpace, types.ResourceSpace]:
    """
    Returns load and workload variance associated to each physical resource.

    :param load_wl: Sorted load associated to workload vectors.
    :param sigma_2_wl: Vector with diagonal of the asymptotic covariance matrix of the workload
        process.
    :param nu: Matrix of sensitivity vectors.
    :return: (load_ph, sigma_2_ph)
        - load_ph: vector with load for each physical resource, i.e. for each row of the
            constituency matrix.
        - sigma_2_ph: vector with asymptotic variance of the workload process for each
            physical resource, i.e. for each row of the constituency matrix.
    """
    assert sigma_2_wl.shape == load_wl.shape
    assert nu.shape[0] == sigma_2_wl.shape[0]

    # Assert that each row of nu lies in the probability simplex.
    np.testing.assert_almost_equal(np.sum(nu, axis=1), np.ones((nu.shape[0],)))

    num_resources = nu.shape[1]
    # Initialise output
    load_ph = np.zeros((num_resources, 1))
    sigma_2_ph = np.zeros((num_resources, 1))
    w_dirs_to_resources = defaultdict(set)

    for s in range(num_resources):
        for i,(load_wl_i, sigma_2_wl_i, nu_i) in enumerate(zip(load_wl, sigma_2_wl, nu)):
            if nu_i[s] > 0:
                load_ph[s] = max(load_ph[s], load_wl_i)
                sigma_2_ph[s] = max(sigma_2_ph[s], sigma_2_wl_i)
                w_dirs_to_resources[i].add(s)

    assert np.all(load_ph >= 0)
    assert np.all(sigma_2_ph >= 0)
    return load_ph, sigma_2_ph, w_dirs_to_resources


def obtain_safety_stock_per_resource(theta: float, load_ph_s: float, sigma_2_ph_s: float,
                                     state: np.ndarray) -> float:
    """
    Returns safety stock value for a given resource as a parametric function of the state, with
    parameter theta. The function has logarithmic form, is increasing with the asymptotic covariance
    of the workload process (sigma_s) and with the load, both associated to the given resource.

    :param theta: tuning hyper-parameter for family of log safety stock functions.
    :param load_ph_s: load for physical resource 's'.
    :param sigma_2_ph_s: asymptotic variance for physical resource 's'.
    :param state: current state, i.e. buffer levels.
    :return tau_s: safety stock threshold (i.e. scalar value) for the s-th resource.
    """
    assert theta > 0
    assert sigma_2_ph_s >= 0
    assert 0 <= load_ph_s < 1
    assert np.all(state >= 0)
    tau_s = theta * math.log(1 + sigma_2_ph_s * np.sum(state) / (theta * (1 - load_ph_s)))
    return tau_s


def obtain_penalty_gradient_per_condition_resource(kappa_s: float, tau_s: float,
                                                   boundary_condition_sj: np.ndarray,
                                                   state: np.ndarray) -> np.ndarray:
    """Returns approximation to the gradient of the penalty function.

    :param kappa_s: penalty coefficient for s-th resource.
    :param tau_s: threshold value for s-th resource.
    :param boundary_condition_sj: j-th row of the boundary condition matrix for the s-th resource.
    :param state: buffer levels.
    :return penalty_grad_sj: gradient of the penalty function for the j-th condition for the s-th
        resource.
    """
    assert boundary_condition_sj.size == state.shape[0]
    grad_sj = np.max((0, tau_s - boundary_condition_sj @ state)) * boundary_condition_sj
    penalty_grad_sj = - kappa_s * grad_sj
    return penalty_grad_sj[:, None]  # Return vector with shape (num_buffers, 1)


def obtain_safety_stock_vector(theta: np.ndarray,
                               load_ph: np.ndarray,
                               sigma_2_ph: np.ndarray,
                               state: np.ndarray,
                               debug_info: bool = False):
    """
    Returns vector whose components are the safety stock per resource.

    :param theta: tuning parameter.
    :param load_ph: vector of loads for each physical resource, with components ordered as the
        constituency matrix.
    :param sigma_2_ph: asymptotic variance of the workload process for each physical resource, with
        components ordered as the constituency matrix.
    :param state: current state, i.e., buffer levels.
    :param debug_info: Boolean flag that indicates whether printing useful debug info.
    :return tau: Vector of safety stocks.
    """
    num_resources = load_ph.shape[0]
    tau = np.zeros((num_resources, 1))
    sigma_2_ph_min = np.maximum(1, sigma_2_ph)
    for s, (load_s, theta_s, sigma_2_s) in enumerate(zip(load_ph, theta, sigma_2_ph_min)):
        tau[s] = obtain_safety_stock_per_resource(theta_s, load_s, sigma_2_s, state)

    if debug_info:
        print(f"Safety stock tau: {np.squeeze(np.round(tau))}")
    return tau
