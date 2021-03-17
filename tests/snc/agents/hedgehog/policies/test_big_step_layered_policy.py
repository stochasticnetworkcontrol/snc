import numpy as np

from snc.agents.hedgehog.params import BigStepLayeredPolicyParams, BigStepPenaltyPolicyParams
from snc.agents.hedgehog.policies.big_step_layered_policy import BigStepLayeredPolicy
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
from snc.agents.hedgehog.workload import workload
import snc.environments.scenarios as scenarios


def get_simple_link_constrained_model():
    cost_per_buffer = np.array([3, 1, 3, 1.5, 3]).reshape(-1, 1)
    param_overrides = dict(alpha1=4.8, mu12=2., mu13=4., mu25=2., mu32=4.5, mu34=1.8,
                           mu35=2., mu45=1., mu5=7.,
                           cost_per_buffer=cost_per_buffer)
    _, env = scenarios.load_scenario('simple_link_constrained_model', 0, param_overrides)

    _, workload_mat, nu = workload.compute_load_workload_matrix(env,6)
    env.workload_mat = workload_mat
    env.nu = nu

    return env


def get_layered_policy_object_for_simple_link_constrained_model(env):
    _, workload_mat, nu = workload.compute_load_workload_matrix(env, 6)
    policy_params = BigStepLayeredPolicyParams('cvx.CPLEX')
    policy_obj = BigStepLayeredPolicy(env.cost_per_buffer,
                                      env.constituency_matrix,
                                      env.job_generator.demand_rate,
                                      env.job_generator.buffer_processing_matrix,
                                      env.workload_mat,
                                      env.nu,
                                      env.list_boundary_constraint_matrices,
                                      env.ind_surplus_buffers,
                                      policy_params)

    return policy_obj


def get_penalty_policy_object_for_simple_link_constrained_model(env, kappa_w, kappa):
    _, workload_mat, nu = workload.compute_load_workload_matrix(env, 6)
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)
    policy_obj = BigStepPolicy(env.cost_per_buffer, 
                               env.constituency_matrix,
                               env.job_generator.demand_rate,
                               env.job_generator.buffer_processing_matrix,
                               env.workload_mat, 
                               env.nu,
                               env.list_boundary_constraint_matrices,
                               env.ind_surplus_buffers,
                               policy_params)

    return policy_obj


def test_no_safety_stocks_stealing_layered_policy():
    env = get_simple_link_constrained_model()

    # system is at high initial state, critical safety stock level in buffer 2 should been
    # maintained
    state = np.array([0, 1000, 0, 0, 0]).reshape(-1, 1)
    horizon = 100
    safety_stocks_vec = np.array([10, 10, 10, 10, 10, 10, 10, 10]).reshape(-1, 1)

    # safety stock penalty is higher than nonidling one
    policy_obj = get_penalty_policy_object_for_simple_link_constrained_model(env=env,
                                                                             kappa_w=1e3,
                                                                             kappa=1e6)

    z_star, _ = policy_obj.get_policy(state=state,
                                      safety_stocks_vec=safety_stocks_vec,
                                      k_idling_set=np.array([]),
                                      draining_bottlenecks=set([0]),
                                      horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # critical safety stock is maintained
    np.testing.assert_almost_equal(new_state[2], 10)

    # safety stock penalty is lower than nonidling one
    policy_obj = get_penalty_policy_object_for_simple_link_constrained_model(env=env,
                                                                             kappa_w=1e6,
                                                                             kappa=1e3 )

    z_star, _ = policy_obj.get_policy(state=state,
                                      safety_stocks_vec=safety_stocks_vec,
                                      k_idling_set=np.array([]),
                                      draining_bottlenecks=set([0]),
                                      horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # critical safety stock level is not maintained
    assert new_state[2] < 5

    # now create layered policy which has no penalty coefficients
    policy_obj = get_layered_policy_object_for_simple_link_constrained_model(env=env)

    z_star, _ = policy_obj.get_policy(state=state,
                                      safety_stocks_vec=safety_stocks_vec,
                                      k_idling_set=np.array([]),
                                      draining_bottlenecks=set([0]),
                                      horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # critical safety stock is maintained
    np.testing.assert_almost_equal(new_state[2], 10)


def test_full_draining_layered_policy():
    env = get_simple_link_constrained_model()

    # system is in drained state and fluid policy should maintain it drained
    state = np.array([0, 0, 0, 0, 0]).reshape(-1, 1)
    horizon = 100
    safety_stocks_vec = np.array([10, 10, 10, 10, 10, 10, 10, 10]).reshape(-1, 1)

    # safety stock penalty is lower than nonidling one
    policy_obj = get_penalty_policy_object_for_simple_link_constrained_model(env=env,
                                                                             kappa_w=1e6,
                                                                             kappa=1e3 )

    z_star,_ = policy_obj.get_policy(state=state,
                                     safety_stocks_vec=safety_stocks_vec,
                                     k_idling_set=np.array([]),
                                     draining_bottlenecks=set([0]),
                                     horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # safety stock level is drained
    np.testing.assert_almost_equal(new_state[2], 0)

    # safety stock penalty is higher than nonidling one
    policy_obj = get_penalty_policy_object_for_simple_link_constrained_model(env=env,
                                                                             kappa_w=1e3,
                                                                             kappa=1e6 )

    z_star,_ = policy_obj.get_policy(state=state,
                                     safety_stocks_vec=safety_stocks_vec,
                                     k_idling_set=np.array([]),
                                     draining_bottlenecks=set([0]),
                                     horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # safety stock is not drained
    np.testing.assert_almost_equal(new_state[2], 10)

    # now create layered policy which has no penalty coeficients
    policy_obj = get_layered_policy_object_for_simple_link_constrained_model(env=env)

    z_star,_ = policy_obj.get_policy(state=state,
                                     safety_stocks_vec=safety_stocks_vec,
                                     k_idling_set=np.array([]),
                                     draining_bottlenecks=set([0]),
                                     horizon=horizon)

    new_state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon

    # safety stock is drained
    np.testing.assert_almost_equal(new_state[2], 0)


def test_backward_fluid_rates_feasibility_drained_state():
    env = get_simple_link_constrained_model()

    # system is in drained state and fluid policy should maintain it drained
    state = np.array([0,0,0,0,0]).reshape(-1,1)
    horizon = 100
    safety_stocks_vec = np.array([10,10,10,10,10,10,10,10]).reshape(-1,1)

    policy_obj = get_layered_policy_object_for_simple_link_constrained_model(env=env)

    for _ in range(10):
        z_star,_ = policy_obj.get_policy(state=state,
                                         safety_stocks_vec=safety_stocks_vec,
                                         k_idling_set=np.array([]),
                                         draining_bottlenecks=set([0]),
                                         horizon=horizon)

        assert np.all([c.value() for c in policy_obj._cost_lp_constraints])

        policy_obj._z_drain.value = z_star
        assert np.all([c.value() for c in policy_obj._draining_lp_constraints])

        policy_obj._z_safe.value = z_star
        assert np.all([c.value() for c in policy_obj._safety_stocks_lp_constraints])

        policy_obj._z_nonidle.value = z_star
        assert np.all([c.value() for c in policy_obj._nonidling_lp_constraints])

        state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon


def test_backward_fluid_rates_feasibility_effective_state():
    env = get_simple_link_constrained_model()

    # system is in drained state and fluid policy should maintain it drained
    state = np.array([0, 1000, 0, 0, 0]).reshape(-1, 1)
    horizon = 100
    safety_stocks_vec = np.array([10, 10, 10, 10, 10, 10, 10, 10]).reshape(-1, 1)

    policy_obj = get_layered_policy_object_for_simple_link_constrained_model(env=env)

    for _ in range(10):
        z_star, _ = policy_obj.get_policy(state=state,
                                          safety_stocks_vec=safety_stocks_vec,
                                          k_idling_set=np.array([]),
                                          draining_bottlenecks=set([0]),
                                          horizon=horizon)

        assert np.all([c.value() for c in policy_obj._cost_lp_constraints])

        policy_obj._z_drain.value = z_star
        assert np.all([c.value() for c in policy_obj._draining_lp_constraints])

        policy_obj._z_safe.value = z_star
        assert np.all([c.value() for c in policy_obj._safety_stocks_lp_constraints])

        policy_obj._z_nonidle.value = z_star
        assert np.all([c.value() for c in policy_obj._nonidling_lp_constraints])

        state = state + (policy_obj.buffer_processing_matrix @ z_star
                         + policy_obj.demand_rate) * horizon
