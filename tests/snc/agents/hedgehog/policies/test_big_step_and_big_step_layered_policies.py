import numpy as np
import pytest

from src.snc import BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams
from src.snc.agents.hedgehog.policies import policy_utils
from src.snc import BigStepLayeredPolicy
from src.snc import BigStepPolicy
from src.snc.agents.hedgehog.policies.big_step_surplus_layered_policy import BigStepSurplusLayeredPolicy
from tests.snc.agents.hedgehog.policies.utils_test_policies import get_allowed_activities


@pytest.fixture(params=["push", "pull"])
def model_type(request):
    return request.param


@pytest.fixture(params=[BigStepLayeredPolicy, BigStepPolicy, BigStepSurplusLayeredPolicy])
def policy_class(request):
    return request.param


def get_policy_params(policy_class):
    if isinstance(policy_class, (BigStepLayeredPolicy, BigStepSurplusLayeredPolicy)):
        return BigStepLayeredPolicyParams(convex_solver='cvx.CPLEX')
    else:
        return BigStepPenaltyPolicyParams(
            convex_solver='cvx.CPLEX',
            boolean_action_flag=False,
            nonidling_penalty_coeff=1e2,
            safety_penalty_coeff=1e2
        )


def test_with_no_tolerance_policy_when_feasible_for_push_and_pull_models(
        model_type,
        policy_class
):
    np.random.seed(42)
    num_buffers = 4
    num_activities = 5
    num_wl_vec = 4
    constituency_matrix = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 1, 1]])
    list_boundary_constraint_matrices = [
        np.array([[1, 1, 0, 0]]),
        np.array([[0, 1, 1, 0]]),
        np.array([[0, 0, 1, 1]])
    ]
    # We add large demand rate to satisfy the extra constraint of nonnegative future state, not
    # present in 'feedback_policy_nonidling_constraint_scipy'.
    demand_rate = 1e6 * np.ones((num_buffers, 1))
    nu = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]])
    k_idling_set = np.array([0, 2])
    draining_bottlenecks = {1}
    horizon = 100
    num_problems = 150
    num_resources = constituency_matrix.shape[0]

    policy_params = get_policy_params(policy_class)

    i = 0  # Num of feasible problems
    diff_z_star = 0
    diff_opt_val = 0

    for _ in range(num_problems):
        state = 20 * np.random.random_sample((num_buffers, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1))
        safety_stocks_vec = 2 * np.random.random_sample((num_resources, 1))
        buffer_processing_matrix = 20 * (
                    np.random.random_sample((num_buffers, num_activities)) - 0.5)
        workload_mat = np.random.random_sample((num_wl_vec, num_buffers))

        if model_type == "push":
            demand_plan = None
            ind_surplus_buffers = None
        else:
            ind_surplus_buffers = [0, 1]
            demand_plan = dict()
            for j in ind_surplus_buffers:
                demand_plan[j] = np.random.choice(np.arange(1, 5))

        kwargs_init = {
            'cost_per_buffer': cost_per_buffer,
            'constituency_matrix': constituency_matrix,
            'demand_rate': demand_rate,
            'buffer_processing_matrix': buffer_processing_matrix,
            'workload_mat': workload_mat,
            'nu': nu,
            'list_boundary_constraint_matrices': list_boundary_constraint_matrices,
            'policy_params': policy_params
        }
        if policy_class != BigStepLayeredPolicy:
            kwargs_init['ind_surplus_buffers'] = ind_surplus_buffers

        policy_cvx = policy_class(**kwargs_init)

        allowed_activities = get_allowed_activities(policy_cvx, num_wl_vec, k_idling_set)

        z_star_cons_cvx, opt_val_cons_cvx = policy_utils.feedback_policy_nonidling_constraint_cvx(
            state,
            cost_per_buffer,
            constituency_matrix,
            buffer_processing_matrix,
            workload_mat,
            safety_stocks_vec,
            k_idling_set,
            draining_bottlenecks,
            list_boundary_constraint_matrices,
            allowed_activities,
            demand_rate,
            horizon,
            demand_plan,
            policy_params.convex_solver
        )

        if z_star_cons_cvx is not None:
            kwargs_get_policy = {
                'state': state,
                'safety_stocks_vec': safety_stocks_vec,
                'k_idling_set': k_idling_set,
                'draining_bottlenecks': draining_bottlenecks,
                'horizon': horizon,
            }
            if policy_class != BigStepLayeredPolicy:
                kwargs_get_policy['demand_plan'] = demand_plan

            z_star_approx_cvx, opt_val_approx_cvx = policy_cvx.get_policy(**kwargs_get_policy)

            opt_val_layer = (cost_per_buffer.T @ buffer_processing_matrix @ z_star_approx_cvx)[0]
            diff_opt_val_i = np.abs((opt_val_cons_cvx - opt_val_layer) / opt_val_cons_cvx)
            diff_z_star_i = np.linalg.norm(z_star_cons_cvx - np.squeeze(z_star_approx_cvx))
            # print("problem:", i, "diff_opt_val=", diff_opt_val_i)
            diff_opt_val += diff_opt_val_i
            diff_z_star += diff_z_star_i
            i += 1

    rel_diff_opt_val = diff_opt_val / i
    rel_diff_z_star = diff_z_star / i
    assert i > 40
    print("======================================")
    print("model_type", model_type)
    print("relative error opt_val=", rel_diff_opt_val)
    print("absolute error z_star=", rel_diff_z_star, "\n")
    print("Num feasible problems with hard constraint: ", i)
    np.testing.assert_almost_equal(rel_diff_opt_val, 0)
