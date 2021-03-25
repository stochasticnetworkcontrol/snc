import numpy as np
import pytest

from snc.agents.hedgehog.params import BigStepPenaltyPolicyParams
from snc.agents.hedgehog.policies import policy_utils
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
from snc.demand_planning.constant_demand_plan import ConstantDemandPlan
from tests.snc.agents.hedgehog.policies.utils_test_policies import get_allowed_activities


def test_obtain_nonidling_resources_empty_botlenecks():
    k_idling_set = np.array([0])
    with pytest.raises(AssertionError):
        _ = policy_utils.obtain_nonidling_bottleneck_resources(0, k_idling_set)


def test_obtain_nonidling_resources_empty_idling_set():
    num_bottlenecks = 3
    k_idling_set = np.array([])
    nonidling_resources = policy_utils.obtain_nonidling_bottleneck_resources(num_bottlenecks,
                                                                             k_idling_set)
    assert np.all(nonidling_resources == np.array(range(num_bottlenecks)))


def test_obtain_nonidling_resources_equal_bottleneck_and_idling_set():
    num_bottlenecks = 3
    k_idling_set = np.array([0, 1, 2])
    nonidling_resources = policy_utils.obtain_nonidling_bottleneck_resources(num_bottlenecks,
                                                                             k_idling_set)
    assert nonidling_resources.size == 0


def test_obtain_nonidling_resources_remove_2_elements():
    num_bottlenecks = 4
    k_idling_set = np.array([1, 2])
    nonidling_resources = policy_utils.obtain_nonidling_bottleneck_resources(num_bottlenecks,
                                                                             k_idling_set)
    assert np.all(nonidling_resources == np.array([0, 3]))


def test_add_draining_bottlenecks_to_nonidling_resources_empty_draining():
    draining_bottlenecks = {}
    nonidling_res = np.array([1, 2])
    nonidling_res = policy_utils.add_draining_bottlenecks_to_nonidling_resources(
        draining_bottlenecks, nonidling_res)
    assert np.all(nonidling_res == np.array([1, 2]))


def test_add_draining_bottlenecks_to_nonidling_resources_both_empty():
    draining_bottlenecks = {}
    nonidling_res = np.array([])
    nonidling_res = policy_utils.add_draining_bottlenecks_to_nonidling_resources(
        draining_bottlenecks, nonidling_res)
    assert nonidling_res.size == 0


def test_add_draining_bottlenecks_to_nonidling_resources_nonidling_res_empty():
    draining_bottlenecks = {0, 3}
    nonidling_res = np.array([])
    nonidling_res = policy_utils.add_draining_bottlenecks_to_nonidling_resources(
        draining_bottlenecks, nonidling_res)
    assert nonidling_res.tolist() == [0, 3]


def test_add_draining_bottlenecks_to_nonidling_resources():
    draining_bottlenecks = {0, 3}
    nonidling_res = np.array([1, 2])
    nonidling_res = policy_utils.add_draining_bottlenecks_to_nonidling_resources(
        draining_bottlenecks, nonidling_res)
    assert nonidling_res.tolist() == [1, 2, 0, 3]


def test_add_draining_bottlenecks_to_nonidling_resources_repeated():
    draining_bottlenecks = {0, 3}
    nonidling_res = np.array([0, 1, 2])
    nonidling_res = policy_utils.add_draining_bottlenecks_to_nonidling_resources(
        draining_bottlenecks, nonidling_res)
    assert nonidling_res.tolist() == [0, 1, 2, 3]


def test_get_index_actions_that_can_drain_buffer_no_action_can_drain_zeros():
    buffer_processing_matrix = np.zeros((3, 3))
    b = 0
    a = policy_utils.get_index_activities_that_can_drain_buffer(b, buffer_processing_matrix)
    assert a.size == 0


def test_get_index_actions_that_can_drain_buffer_no_action_can_drain_positive():
    buffer_processing_matrix = np.ones((3, 3))
    b = 0
    a = policy_utils.get_index_activities_that_can_drain_buffer(b, buffer_processing_matrix)
    assert a.size == 0


def test_get_index_actions_that_can_drain_buffer():
    buffer_processing_matrix = np.array([[1, -1, 0], [-1, -1, 0], [0, 0, 1]])
    b = 0
    a = policy_utils.get_index_activities_that_can_drain_buffer(b, buffer_processing_matrix)
    assert a == np.array([1])

    b = 1
    a = policy_utils.get_index_activities_that_can_drain_buffer(b, buffer_processing_matrix)
    assert np.all(a == np.array([0, 1]))


def test_obtain_general_empty_buffers_constraints_matrix_form_larger_state_than_num_activities():
    state = np.array([4, 4, 4])[:, None]
    buffer_processing_matrix = - np.ones((3, 3))
    a_mat, b_vec = policy_utils.obtain_general_empty_buffers_constraints_matrix_form(
        state, buffer_processing_matrix)
    assert a_mat == []
    assert b_vec == []


def test_obtain_general_empty_buffers_constraints_matrix_form_equal_state_than_num_activities():
    state = np.array([1, 2, 3])[:, None]
    buffer_processing_matrix = np.array([[-1, 0, 0], [1, -1, 0], [-1, -1, -1]])
    a_mat, b_vec = policy_utils.obtain_general_empty_buffers_constraints_matrix_form(
        state, buffer_processing_matrix)
    assert a_mat == []
    assert b_vec == []


def test_obtain_general_empty_buffers_constraints_matrix_empty_buffer():
    state = np.array([0, 2, 3])[:, None]
    buffer_processing_matrix = np.array([[-1, 0, 0], [-1, -1, 0], [-1, -1, -1]])
    a_mat, b_vec = policy_utils.obtain_general_empty_buffers_constraints_matrix_form(
        state, buffer_processing_matrix)
    assert np.all(a_mat == np.array([1, 0, 0]))
    assert np.all(b_vec == np.array([0]))


def test_obtain_general_empty_buffers_constraints_matrix_two_empty_buffers():
    state = np.array([0, 0, 3])[:, None]
    buffer_processing_matrix = np.array([[-1, 0, 0], [-1, -1, 0], [-1, -1, -1]])
    a_mat, b_vec = policy_utils.obtain_general_empty_buffers_constraints_matrix_form(
        state, buffer_processing_matrix)
    assert np.all(a_mat[0] == np.array([1, 0, 0]))
    assert np.all(a_mat[1] == np.array([1, 1, 0]))
    assert np.all(b_vec[0] == np.array([0]))
    assert np.all(b_vec[1] == np.array([0]))


def test_obtain_general_empty_buffers_constraints_matrix_routing():
    state = np.array([0, 1, 3])[:, None]
    buffer_processing_matrix = np.array([[-1, 0, 0], [-1, -1, 0], [-1, -1, -1]])
    a_mat, b_vec = policy_utils.obtain_general_empty_buffers_constraints_matrix_form(
        state, buffer_processing_matrix)
    assert np.all(a_mat[0] == np.array([1, 0, 0]))
    assert np.all(a_mat[1] == np.array([1, 1, 0]))
    assert np.all(b_vec[0] == np.array([0]))
    assert np.all(b_vec[1] == np.array([1]))


def test_get_index_non_exit_actions_no_exit():
    buffer_procesing_matrix = np.ones((3, 3))
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_procesing_matrix)
    assert ind_non_exit_actions == [0, 1, 2]


def test_get_index_non_exit_actions_one_action_typical_push():
    buffer_procesing_matrix = np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1]])
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_procesing_matrix)
    assert ind_non_exit_actions == [0, 1]


def test_get_index_non_exit_actions_two_actions_push():
    buffer_procesing_matrix = np.array([[-1, 0, 0, -1], [1, -1, 0, 0], [0, 1, -1, 0]])
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_procesing_matrix)
    assert ind_non_exit_actions == [0, 1]


def test_get_index_non_exit_actions_one_action_typical_demand():
    buffer_procesing_matrix = np.array([[-1, 0, 1],
                                        [1, -1, 0],
                                        [0, -1, 1]])
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_procesing_matrix)
    assert ind_non_exit_actions == [0, 2]


def test_get_index_non_exit_actions_two_actions_demand():
    buffer_procesing_matrix = np.array([[-1, 0, 1, -1],
                                        [1, -1, 0, 0],
                                        [0, -1, 1, -1]])
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_procesing_matrix)
    assert ind_non_exit_actions == [0, 2]


def test_get_index_non_exit_actions_diag_buffer_processing_mat():
    """ This happens in the input_queued_switch_3x3_model."""
    buffer_processing_matrix = - np.diag(np.ones((4, )))
    ind_non_exit_actions = policy_utils.get_index_non_exit_activities(buffer_processing_matrix)
    assert ind_non_exit_actions == []


def test_feedback_policy_nonidling_constraint_two_alternative_methods():
    np.random.seed(42)
    num_buffers = 4
    num_activities = 5
    num_wl_vec = 4
    constituency_matrix = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
    nu = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]])
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 1]]), np.array([[0, 1, 1, 0]])]
    ind_surplus_buffers = None
    kappa_w = 1e2
    kappa = 1e2
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)
    horizon = 100
    k_idling_set = np.array(range(num_buffers))
    draining_bottlenecks = {0}
    num_problems = 100

    i = 0  # Num of feasible problems
    diff_opt_val = 0
    diff_z_star = 0
    num_resources = constituency_matrix.shape[0]
    for _ in range(num_problems):
        state = 10 * np.random.random_sample((num_buffers, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1)) + 0.1
        safety_stocks_vec = 0 * np.random.random_sample((num_resources, 1))
        buffer_processing_matrix = 20 * (np.random.random_sample((num_buffers, num_activities)) - 1)
        workload_mat = 0.5 * np.random.random_sample((num_wl_vec, num_buffers))
        demand_rate = np.random.random_sample((num_buffers, 1))

        policy_cvx = BigStepPolicy(
            cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
            workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
        allowed_activities = get_allowed_activities(policy_cvx, num_wl_vec, k_idling_set)
        z_star_scipy, opt_val_scipy = \
            policy_utils.feedback_policy_nonidling_constraint_scipy(
                state, cost_per_buffer, constituency_matrix, buffer_processing_matrix, workload_mat,
                safety_stocks_vec, k_idling_set, draining_bottlenecks,
                list_boundary_constraint_matrices, allowed_activities, demand_rate, horizon)

        z_star_cvx, opt_val_cvx = \
            policy_utils.feedback_policy_nonidling_constraint_cvx(
                state, cost_per_buffer, constituency_matrix, buffer_processing_matrix, workload_mat,
                safety_stocks_vec, k_idling_set, draining_bottlenecks,
                list_boundary_constraint_matrices, allowed_activities, demand_rate, horizon)

        if z_star_scipy is not None and z_star_cvx is not None:
            z_star_scipy = z_star_scipy[:, None]

            diff_opt_val_i = np.abs((opt_val_cvx - opt_val_scipy) / opt_val_cvx)
            diff_z_star_i = np.linalg.norm(z_star_cvx - z_star_scipy)
            print(i, "diff opt val=", diff_opt_val_i)
            diff_opt_val += diff_opt_val_i
            diff_z_star += diff_z_star_i
            i += 1

            # np.testing.assert_allclose(z_star_scipy, z_star_cvx, rtol=1e-2)
            # np.testing.assert_allclose(opt_val_scipy, opt_val_cvx, rtol=1e-2)

    rel_diff_opt_val = diff_opt_val / i
    rel_diff_z_star = diff_z_star / i
    np.testing.assert_almost_equal(rel_diff_opt_val, 0, decimal=2)
    np.testing.assert_almost_equal(rel_diff_z_star, 0, decimal=2)

    print("======================================")
    print("relative error opt_val=", rel_diff_opt_val)
    print("absolute error z_star=", rel_diff_z_star, "\n")
    print("Num feasible problems: ", i)
