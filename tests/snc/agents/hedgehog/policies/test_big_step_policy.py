import pytest
import numpy as np

from snc.agents.hedgehog.params import BigStepPenaltyPolicyParams
from snc.agents.hedgehog.workload import workload
from snc.agents.hedgehog.policies import policy_utils
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
import snc.environments.examples as examples
import snc.utils.snc_tools as snc_tools
from tests.snc.agents.hedgehog.policies.utils_test_policies import get_allowed_activities


def get_cost(cost_per_buffer, buffer_processing_matrix, z_star):
    return cost_per_buffer.T @ buffer_processing_matrix @ z_star


def test_get_forbidden_actions_per_bottleneck_loop_2_queues():
    env = examples.loop_2_queues(mu1=1, mu12=0.5, mu2=0.8, mu21=1,
                                 cost_per_buffer=np.array([[1], [0.1]]),
                                 demand_rate=np.array([[0.1], [0.95]]))
    load, workload_mat, nu = workload.compute_load_workload_matrix(env)
    xi_s = workload_mat[0, :]  # Pooled resource.
    nu_s = nu[0, :]  # Pooled resource.
    fa_s = BigStepPolicy.get_forbidden_activities_per_bottleneck(
        xi_s, nu_s, env.job_generator.buffer_processing_matrix, env.constituency_matrix)
    assert fa_s == [1]  # We cannot send stuff back through mu12.


def test_get_policy_non_empty_buffers():
    """
    We have a serial line of two resources with one buffer each:
            demand --> q_1 + resource_1 --> q_2 + resource_2 -->
    We relax the nonidling constraint for first resource. Since the cost in the second resource is
    much higher, the LP will make the first resource to idle.
    """
    k_idling_set = np.array([0])
    draining_bottlenecks = {}

    kappa_w = 1e2  # Very high to force the influence of the constraint
    kappa = 1e2
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    state = np.array([[10], [10]])
    c_1 = 1
    c_2 = 100
    mu_1 = 100
    mu_2 = 1
    alpha = 0.9
    cost_per_buffer = np.array([[c_1], [c_2]])
    demand_rate = np.array([[alpha], [0]])
    buffer_processing_matrix = np.array([[-mu_1, 0], [mu_1, -mu_2]])
    constituency_matrix = np.eye(2)
    safety_stocks_vec = np.zeros((2, 1))
    list_boundary_constraint_matrices = []
    ind_surplus_buffers = None

    workload_mat = np.array([[1 / mu_1, 0], [1 / mu_2, 1 / mu_2]])
    nu = np.eye(2)
    horizon = 100

    policy = BigStepPolicy(cost_per_buffer, constituency_matrix, demand_rate,
                           buffer_processing_matrix, workload_mat, nu,
                           list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star, _ = policy.get_policy(state, **kwargs)
    np.testing.assert_allclose(z_star, np.array([[0], [1]]), rtol=1e-2, atol=1e-2)


def test_get_policy_with_two_alternative_methods():
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
    ind_surplus_buffers = None
    num_resources = constituency_matrix.shape[0]
    k_idling_set = np.array([0])
    draining_bottlenecks = {1}
    nu = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]])
    kappa_w = 1e2
    kappa = 1e2
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)
    for _ in range(100):
        state = 10 * np.random.random_sample((num_buffers, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1))
        demand_rate = 10 * np.random.random_sample((num_buffers, 1))
        safety_stocks_vec = 2 * np.random.random_sample((num_resources, 1))
        buffer_processing_matrix = 20 * (np.random.random_sample((num_buffers, num_activities)) - .5)
        workload_mat = np.random.random_sample((num_wl_vec, num_buffers))
        horizon = 100

        policy_cvx = BigStepPolicy(
            cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
            workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
        kwargs = {
            'safety_stocks_vec': safety_stocks_vec,
            'k_idling_set': k_idling_set,
            'draining_bottlenecks': draining_bottlenecks,
            'horizon': horizon
        }
        z_star_pen_cvx, opt_val_pen_cvx = policy_cvx.get_policy(state, **kwargs)

        allowed_activities = get_allowed_activities(policy_cvx, num_wl_vec, k_idling_set)
        z_star_pen_scipy, opt_val_pen_scipy = policy_utils.feedback_policy_nonidling_penalty_scipy(
            state, cost_per_buffer, constituency_matrix, buffer_processing_matrix, workload_mat,
            safety_stocks_vec, k_idling_set, draining_bottlenecks, kappa_w,
            list_boundary_constraint_matrices, allowed_activities,
            method='revised simplex', demand_rate=demand_rate, horizon=horizon)

        cvx_cost = get_cost(cost_per_buffer, buffer_processing_matrix, z_star_pen_cvx)
        scipy_cost = get_cost(cost_per_buffer, buffer_processing_matrix, z_star_pen_scipy)
        np.testing.assert_almost_equal(np.squeeze(cvx_cost), scipy_cost, decimal=3)


def test_get_policy_with_two_alternative_methods_when_empty_buffers():
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
    ind_surplus_buffers = None
    k_idling_set = np.array([0])
    draining_bottlenecks = {1}
    nu = np.array([[0.5, 0, 0.5], [0, 1, 0], [0, 0, 1], [0, 0.5, 0.5]])
    horizon = 1
    kappa_w = 1e2
    kappa = 1e2
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)
    num_resources = constituency_matrix.shape[0]
    for _ in range(100):
        state = np.zeros((num_buffers, 1))
        demand_rate = np.random.random_sample((num_buffers, 1))
        state[np.random.randint(0, num_buffers, 2)] = 10 * np.random.random_sample(size=(2, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1))
        safety_stocks_vec = np.zeros((num_resources, 1))
        buffer_processing_matrix = 20 * (np.random.random_sample((num_buffers, num_activities)) - .5)
        workload_mat = np.random.random_sample((num_wl_vec, num_buffers))

        policy_cvx = BigStepPolicy(
            cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
            workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
        kwargs = {
            'safety_stocks_vec': safety_stocks_vec,
            'k_idling_set': k_idling_set,
            'draining_bottlenecks': draining_bottlenecks,
            'horizon': horizon
        }
        z_star_pen_cvx, opt_val_pen_cvx = policy_cvx.get_policy(state, **kwargs)

        allowed_activities = get_allowed_activities(policy_cvx, num_wl_vec, k_idling_set)
        z_star_pen_scipy, opt_val_pen_scipy = policy_utils.feedback_policy_nonidling_penalty_scipy(
            state, cost_per_buffer, constituency_matrix, buffer_processing_matrix, workload_mat,
            safety_stocks_vec, k_idling_set, draining_bottlenecks, kappa_w,
            list_boundary_constraint_matrices, allowed_activities,
            method='revised simplex', demand_rate=demand_rate, horizon=horizon)

        cvx_cost = get_cost(cost_per_buffer, buffer_processing_matrix, z_star_pen_cvx)
        scipy_cost = get_cost(cost_per_buffer, buffer_processing_matrix, z_star_pen_scipy)
        np.testing.assert_almost_equal(np.squeeze(cvx_cost), scipy_cost, decimal=3)


@pytest.mark.parametrize("ex_env", ['input_queued_switch_3x3_model',
                                    'klimov_model',
                                    'ksrs_network_model',
                                    'processor_sharing_model',
                                    'simple_link_constrained_model',
                                    'simple_link_constrained_with_route_scheduling_model',
                                    'simple_reentrant_line_model',
                                    'simple_routing_model',
                                    'single_server_queue',
                                    'three_station_network_model'])
@pytest.mark.parametrize("horizon", [100, int(1e6)])
def test_get_policy_examples_two_alternative_methods_multiple_horizons(
        ex_env, horizon):
    seed = 42
    np.random.seed(seed)

    env = eval('examples.' + ex_env)(job_gen_seed=seed)

    kappa_w = 1e2
    kappa = 1e2
    safety_stocks_vec = 2 * np.ones((env.num_resources, 1))
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    load, workload_mat, nu = workload.compute_load_workload_matrix(env)
    num_wl_vec = workload_mat.shape[0]
    k_idling_set = np.random.randint(0, workload_mat.shape[0], 1)
    draining_bottlenecks = set(np.random.randint(0, workload_mat.shape[0], 1))

    state = 20 * np.ones((env.num_buffers, 1))

    policy_cvx = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star_cvx, opt_val_cvx = policy_cvx.get_policy(state, **kwargs)

    allowed_activities = get_allowed_activities(policy_cvx, num_wl_vec, k_idling_set)
    z_star_scipy, opt_val_scipy = policy_utils.feedback_policy_nonidling_penalty_scipy(
        state, env.cost_per_buffer, env.constituency_matrix,
        env.job_generator.buffer_processing_matrix, workload_mat, safety_stocks_vec,
        k_idling_set, draining_bottlenecks, kappa_w, env.list_boundary_constraint_matrices,
        allowed_activities, method='revised simplex', demand_rate=env.job_generator.demand_rate,
        horizon=horizon)

    # SciPy has safety stock constraints, while CVX has a safety stock penalty.
    opt_val_cvx = (env.cost_per_buffer.T @ env.job_generator.buffer_processing_matrix
                   @ z_star_cvx)[0]
    opt_val_scipy = (env.cost_per_buffer.T @ env.job_generator.buffer_processing_matrix
                   @ z_star_scipy[:, None])[0]
    np.testing.assert_allclose(opt_val_cvx, opt_val_scipy, rtol=5e-2)


def test_big_step_policy_k_idling_set_proper_idle():
    """We have a serial line of two resources with one buffer each:
        demand --> q_1 + resource_1 --> q_2 + resource_2 -->
    We relax the nonidling constraint for first resource. Since the cost in the second resource is
    much higher, the LP will make the first resource to idle."""
    k_idling_set = np.array([0])
    draining_bottlenecks = {}
    kappa_w = 1e2
    kappa = 1e2

    state = np.array([[10], [10]])
    c_1 = 1
    c_2 = 100
    mu_1 = 100
    mu_2 = 1
    cost_per_buffer = np.array([[c_1], [c_2]])
    constituency_matrix = np.eye(2)
    list_boundary_constraint_matrices = [np.array([[1, 0]]), np.array([[0, 1]])]
    ind_surplus_buffers = None
    demand_rate = np.ones((2, 1))
    buffer_processing_matrix = np.array([[-mu_1, 0], [mu_1, -mu_2]])
    workload_mat = np.array([[1 / mu_1, 0], [1 / mu_2, 1 / mu_2]])
    nu = np.eye(2)
    safety_stocks_vec = np.zeros((2, 1))
    horizon = 1
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy = BigStepPolicy(
        cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
        workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star_pen, _ = policy.get_policy(state, **kwargs)
    np.testing.assert_almost_equal(z_star_pen, np.array([[0], [1]]))


def test_big_step_policy_k_idling_set_dont_relax_idling():
    """
    We have a serial line of two resources with one buffer each:
        demand --> q_1 + resource_1 --> q_2 + resource_2 -->
    We don't relax the nonidling constraint for first resource. So even when the cost in the second
    resource is much higher than the first resource, the LP will make the first resource to work.
    """
    k_idling_set = np.array([])
    draining_bottlenecks = {}

    kappa_w = 1e4
    kappa = 1e2

    state = np.array([[10], [10]])
    c_1 = 1
    c_2 = 100
    mu_1 = 100
    mu_2 = 1
    cost_per_buffer = np.array([[c_1], [c_2]])
    constituency_matrix = np.eye(2)
    list_boundary_constraint_matrices = [np.array([[1, 0]]), np.array([[0, 1]])]
    ind_surplus_buffers = None
    demand_rate = np.ones((2, 1))
    buffer_processing_matrix = np.array([[-mu_1, 0], [mu_1, -mu_2]])
    workload_mat = np.array([[1 / mu_1, 0], [1 / mu_2, 1 / mu_2]])
    nu = np.eye(2)
    safety_stocks_vec = np.zeros((2, 1))
    horizon = 1
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy = BigStepPolicy(
        cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
        workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star_pen, _ = policy.get_policy(state, **kwargs)
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(z_star_pen[0], 0)
    np.testing.assert_almost_equal(z_star_pen[1], 1)


def test_big_step_policy_k_idling_set_relax_2nd_resource():
    """
    We have a serial line of two resources with one buffer each:
        demand --> q_1 + resource_1 --> q_2 + resource_2 -->
    We relax the nonidling constraint for second resource. So even when the cost in the second
    resource is much higher than the first resource, the LP will make the first resource to work.
    """
    k_idling_set = np.array([])
    draining_bottlenecks = {1}

    kappa_w = 1e4
    kappa = 1e2

    state = np.array([[10], [10]])
    c_1 = 1
    c_2 = 100
    mu_1 = 100
    mu_2 = 1
    cost_per_buffer = np.array([[c_1], [c_2]])
    constituency_matrix = np.eye(2)
    list_boundary_constraint_matrices = [np.array([[1, 0]]), np.array([[0, 1]])]
    ind_surplus_buffers = None
    demand_rate = np.ones((2, 1))
    buffer_processing_matrix = np.array([[-mu_1, 0], [mu_1, -mu_2]])
    workload_mat = np.array([[1 / mu_1, 0], [1 / mu_2, 1 / mu_2]])
    nu = np.eye(2)
    safety_stocks_vec = np.zeros((2, 1))
    horizon = 1
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy = BigStepPolicy(
        cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
        workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star_pen, _ = policy.get_policy(state, **kwargs)
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(z_star_pen[0], 0)
    np.testing.assert_almost_equal(z_star_pen[1], 1, decimal=5)


@pytest.fixture(params=[(-1, 1, 1), (1, -1, 1), (1, 1, -1)])
def some_vars_fixture(request):
    return request.param


def test_get_policy_negative_kappa_w_kappa_horizon(some_vars_fixture):
    kappa_w, kappa, horizon = some_vars_fixture

    num_buffers = 4
    num_activities = 5
    num_resources = 5
    num_wl_vec = 4
    constituency_matrix = np.eye(num_resources)
    list_boundary_constraint_matrices = [np.array([[1, 0]]), np.array([[0, 1]])]
    ind_surplus_buffers = None
    cost_per_buffer = np.zeros((num_buffers, 1))
    demand_rate = np.zeros((num_buffers, 1))
    buffer_processing_matrix = np.zeros((num_buffers, num_activities))
    workload_mat = np.zeros((num_wl_vec, num_buffers))
    nu = np.zeros(num_resources)
    boolean_action_flag = False,
    convex_solver = 'cvx.CPLEX',
    policy_params = BigStepPenaltyPolicyParams(convex_solver, boolean_action_flag, kappa_w, kappa)

    with pytest.raises(AssertionError):
        _ = BigStepPolicy(
            cost_per_buffer, constituency_matrix, demand_rate, buffer_processing_matrix,
            workload_mat, nu, list_boundary_constraint_matrices, ind_surplus_buffers, policy_params)


def test_assertion_large_horizon_with_boolean_action_constraints():
    horizon = 2
    boolean_action_flag = True
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', boolean_action_flag, 1e2, 1e2)
    policy = BigStepPolicy(np.ones((2, 1)), np.eye(2), np.ones((2, 1)),
                           -1.1 * np.eye(2), np.eye(2), np.eye(2),
                           [np.array([[1, 0]]), np.array([[0, 1]])], None, policy_params)
    kwargs = {
        'safety_stocks_vec': np.zeros((2, 1)),
        'k_idling_set': np.array([]),
        'draining_bottlenecks': {},
        'horizon': horizon
    }
    with pytest.raises(AssertionError):
        policy.get_policy(np.zeros((2, 1)), **kwargs)


def perform_boolean_constraints_simple_reentrant_line_test(state, z_star_true):
    """
    Set parameters such that the objective is:
        f(zeta) = -0.5 * mu1 * zeta_1  + 1 * mu2 * zeta_2 - 2 * mu3 * zeta_3.
    State dynamics constraint is:
        x_1 - mu1 * zeta_1 >= 0
        x_2 + mu1 * zeta_1 - mu2 * zeta_2 >= 0
        x_3 + mu2 * zeta_2 - mu3 * zeta_3 >= 0.
    LP prioritises working on buffer 3 as much as possible. The only motivation to work on
    buffers 1 and 2 is just to satisfy the state dynamics constraint.
    """
    horizon = 1
    boolean_action_flag = True
    cost_per_buffer = np.array([[1.5], [1], [2]])

    env = examples.simple_reentrant_line_model(cost_per_buffer=cost_per_buffer)
    load, workload_mat, nu = workload.compute_load_workload_matrix(env)

    safety_stocks_vec = np.zeros((env.num_resources, 1))
    kappa_w = 0
    kappa = 0
    k_idling_set = np.array([])
    draining_bottlenecks = {}
    policy_params = BigStepPenaltyPolicyParams("cvx.CPLEX", boolean_action_flag, kappa_w, kappa)
    policy_cvx = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star, _ = policy_cvx.get_policy(state, **kwargs)
    assert snc_tools.is_approx_binary(z_star)
    np.testing.assert_almost_equal(z_star, z_star_true)


def test_boolean_constraints_simple_reentrant_line_empty_buffers_2_3():
    """
    Continuous solution would give positive rate to all buffers. Boolean solution only gives to
    buffer 1. The reason being that resource 1 can only work on either buffer 1 or 3. Then, if
    resource 2 works, it will increase the cost since it cannot be drained.
    """
    state = np.array([[1], [0], [0]])
    z_star_true = np.array([[1], [0], [0]])
    perform_boolean_constraints_simple_reentrant_line_test(state, z_star_true)


def test_boolean_constraints_simple_reentrant_line_empty_buffers_1_2():
    """
    Case where continuous and boolean solution are equal.
    """
    state = np.array([[0], [1], [1]])
    z_star_true = np.array([[0], [0], [1]])
    perform_boolean_constraints_simple_reentrant_line_test(state, z_star_true)


def test_boolean_constraints_simple_reentrant_line_empty_buffer_1():
    """
    Since x_3 < mu3, continuous solution would work on buffer 2 to allow work on buffer 3 as much as
    possible (i.e., while still satisfying the constraint), but not more since it would incur extra
    cost (remember that mu2 is positive in the objective). However, the boolean action makes working
    on buffer 2 at full throttle as it still compensates the benefit.
    """
    # Note that state should be integer. The following state is only for testing purposes:
    fake_state = np.array([[0], [1], [0.7]])
    z_star_true = np.array([[0], [1], [1]])
    perform_boolean_constraints_simple_reentrant_line_test(fake_state, z_star_true)


def test_big_step_policy_safety_stock_active():
    """
    Resource is below safety stock threshold. Nonidling penalty is zero. Minimum cost is achieved by
    draining buffer 3. However, resource 1 will drain buffer 1, so buffer 2 reaches its safety stock
    level.
    """
    state = np.array([0, 0, 100])[:, None]
    alpha1 = 8
    mu1 = 20
    mu2 = 10
    mu3 = 20
    safety_stocks_vec = 8 * np.ones((2, 1))
    horizon = 22
    cost_per_buffer = np.array([[1.5], [1], [2]])
    z_star_true = np.array([0.4, 0, 0.6])[:, None]

    env = examples.simple_reentrant_line_model(alpha1, mu1, mu2, mu3, cost_per_buffer)
    load, workload_mat, nu = workload.compute_load_workload_matrix(env)

    kappa_w = 0
    kappa = 1e2
    k_idling_set = np.array([])
    draining_bottlenecks = {}
    boolean_action_flag = False
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', boolean_action_flag, kappa_w, kappa)

    policy_cvx = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)
    kwargs = {
        'safety_stocks_vec': safety_stocks_vec,
        'k_idling_set': k_idling_set,
        'draining_bottlenecks': draining_bottlenecks,
        'horizon': horizon
    }
    z_star, _ = policy_cvx.get_policy(state, **kwargs)
    np.testing.assert_allclose(z_star, z_star_true, rtol=1e-4)
