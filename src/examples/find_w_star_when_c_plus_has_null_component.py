import numpy as np
import matplotlib.pyplot as plt
import src.snc.agents.hedgehog.strategic_idling.strategic_idling as si


def obtain_w_star_for_different_penalty_values(cost_vector, workload_matrix, w, penalty_range):
    w_star_penalties = np.zeros((2, len(penalty_range)))
    for i, penalty in enumerate(penalty_range):
        si_object = si.StrategicIdling(
            workload_mat=workload_matrix, cost_per_buffer=cost_vector,
            strategic_idling_params=si.get_default_strategic_idling_params(
                penalty_coeff_w_star=penalty),
            workload_cov=None, neg_log_discount_factor=None, load=None)
        w_star_penalties[:, [i]] = si_object._find_workload_with_min_eff_cost_by_idling(w)
    return w_star_penalties


def obtain_distance_from_computed_w_star_to_ground_truth(w_star_list, w_star):
    num_elements = w_star_list.shape[1]
    distance = np.nan * np.ones((num_elements, 1))
    for i in range(num_elements):
        distance[i] = np.linalg.norm(w_star_list[:, [i]] - w_star)
    return distance


if __name__ == '__main__':
    c_1 = 2
    c_2 = 1
    c_3 = 2
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    mu_1 = 2
    mu_2 = 1
    mu_3 = 2
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    c_plus = np.array(
        [[mu_1 * (c_1 - c_2)], [mu_2 * (c_2 * (1 + mu_1 / mu_3) - c_1 * mu_1 / mu_3)]])
    c_minus = np.array([[mu_3 * c_3], [mu_2 * (c_1 - c_3 * (1 + mu_3 / mu_1))]])
    psi_plus = c_plus - c_minus

    w0 = np.array([[1], [0.]])  # Got from x = np.array([[0.9], [0], [0.2]])
    w_star_theory = np.array([w0[0], - w0[0] * psi_plus[0] / psi_plus[1]])

    penalty_w_star_range = [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    w_star_lam = obtain_w_star_for_different_penalty_values(cost_per_buffer, workload_mat, w0,
                                                            penalty_w_star_range)
    dist = obtain_distance_from_computed_w_star_to_ground_truth(w_star_lam, w_star_theory)
    print(f"w_star for different penalties:\n {w_star_lam}")
    print(f"distance: {dist.T}")
    plt.plot(penalty_w_star_range, dist)
    plt.xscale("log")
    plt.show()
