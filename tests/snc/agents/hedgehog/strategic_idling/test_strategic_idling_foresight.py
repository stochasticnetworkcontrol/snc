import numpy as np

import snc.environments.examples as examples
import snc.agents.hedgehog.workload.workload as wl
from snc.agents.hedgehog.params import BigStepPenaltyPolicyParams, StrategicIdlingParams
from snc.agents.hedgehog.policies.big_step_policy import BigStepPolicy
from snc.agents.hedgehog.strategic_idling.strategic_idling_foresight import StrategicIdlingForesight


def test_cum_cost_computation_standard_hedging_regime():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68,
                                               cost_per_buffer=np.array([1.5, 1, 2])[:, None])
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    kappa_w = np.sum(env.cost_per_buffer) * 10
    kappa = 1e4
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy_object = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)

    si_object = StrategicIdlingForesight(workload_mat=workload_mat,
                                         neg_log_discount_factor=neg_log_discount_factor,
                                         load=load,
                                         cost_per_buffer=env.cost_per_buffer,
                                         model_type=env.model_type,
                                         policy_object=policy_object,
                                         strategic_idling_params=strategic_idling_params,
                                         workload_cov=workload_cov)

    init_state = np.array([0, 0, 10000])[:, np.newaxis]
    init_w = workload_mat @ init_state

    si_object._current_state = init_state

    cw_vars = si_object._non_negative_workloads(init_w)
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(init_w,
                                                                                        cw_vars)

    gto_cost = si_object._roll_out_gto_fluid_policy(cw_vars)
    std_hedging_cost = si_object._roll_out_hedgehog_fluid_policy(cw_vars)

    np.testing.assert_allclose(std_hedging_cost/1e6, 1967, rtol=0.03)
    np.testing.assert_allclose(gto_cost/1e6, 1967, rtol=0.03)


def test_cum_cost_computation_switching_curve_regime():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.7, mu2=0.34, mu3=0.7,
                                               cost_per_buffer=np.array([1.5, 1, 2])[:, None])
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    kappa_w = np.sum(env.cost_per_buffer) * 10
    kappa = 1e6
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy_object = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)

    si_object = StrategicIdlingForesight(workload_mat=workload_mat,
                                         neg_log_discount_factor=neg_log_discount_factor,
                                         load=load,
                                         cost_per_buffer=env.cost_per_buffer,
                                         model_type=env.model_type,
                                         policy_object=policy_object,
                                         strategic_idling_params=strategic_idling_params,
                                         workload_cov=workload_cov)

    init_state = np.array([0, 0, 10000])[:, np.newaxis]
    init_w = workload_mat @ init_state

    si_object._current_state = init_state

    cw_vars = si_object._non_negative_workloads(init_w)
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(init_w,
                                                                                        cw_vars)

    gto_cost = si_object._roll_out_gto_fluid_policy(cw_vars)
    std_hedging_cost = si_object._roll_out_hedgehog_fluid_policy(cw_vars)

    np.testing.assert_allclose(std_hedging_cost/1e6,1487,rtol=0.03)
    np.testing.assert_allclose(gto_cost/1e6,1750,rtol=0.03)

    si_object.get_allowed_idling_directions(init_state)
    assert si_object._current_regime == "standard_hedging"
    assert si_object._original_target_dyn_bot_set == set([0])


def test_cum_cost_computation_switching_curve_regime_homogenous_cost():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.7, mu2=0.34, mu3=0.7,
                                               cost_per_buffer=np.array([1, 1.001, 1])[:, None])
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    kappa_w = np.sum(env.cost_per_buffer) * 10
    kappa = 1e4
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy_object = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)

    si_object = StrategicIdlingForesight(workload_mat=workload_mat,
                                         neg_log_discount_factor=neg_log_discount_factor,
                                         load=load,
                                         cost_per_buffer=env.cost_per_buffer,
                                         model_type=env.model_type,
                                         policy_object=policy_object,
                                         strategic_idling_params=strategic_idling_params,
                                         workload_cov=workload_cov)

    init_state = np.array([0, 0, 10000])[:, np.newaxis]
    init_w = workload_mat @ init_state

    si_object._current_state = init_state

    cw_vars = si_object._non_negative_workloads(init_w)
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(init_w,
                                                                                        cw_vars)

    gto_cost = si_object._roll_out_gto_fluid_policy(cw_vars)
    std_hedging_cost = si_object._roll_out_hedgehog_fluid_policy(cw_vars)

    np.testing.assert_allclose(std_hedging_cost/1e6,1214,rtol=0.03)
    np.testing.assert_allclose(gto_cost/1e6,950,rtol=0.03)

    si_object.get_allowed_idling_directions(init_state)
    assert si_object._current_regime == "gto"
    assert si_object._original_target_dyn_bot_set == set([0,1])


def test_switch_of_regime_from_gto():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.7, mu2=0.34, mu3=0.7,
                                               cost_per_buffer=np.array([1.5, 1, 2])[:, None])
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    kappa_w = np.sum(env.cost_per_buffer) * 10
    kappa = 1e4
    policy_params = BigStepPenaltyPolicyParams('cvx.CPLEX', False, kappa_w, kappa)

    policy_object = BigStepPolicy(
        env.cost_per_buffer, env.constituency_matrix, env.job_generator.demand_rate,
        env.job_generator.buffer_processing_matrix, workload_mat, nu,
        env.list_boundary_constraint_matrices, env.ind_surplus_buffers, policy_params)

    si_object = StrategicIdlingForesight(workload_mat=workload_mat,
                                         neg_log_discount_factor=neg_log_discount_factor,
                                         load=load,
                                         cost_per_buffer=env.cost_per_buffer,
                                         model_type=env.model_type,
                                         policy_object=policy_object,
                                         strategic_idling_params=strategic_idling_params,
                                         workload_cov=workload_cov)

    init_state = np.array([2500, 0, 3000])[:, np.newaxis]
    init_w = workload_mat @ init_state

    si_object._current_state = init_state

    cw_vars = si_object._non_negative_workloads(init_w)
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(init_w,
                                                                                        cw_vars)

    gto_cost = si_object._roll_out_gto_fluid_policy(cw_vars)
    std_hedging_cost = si_object._roll_out_hedgehog_fluid_policy(cw_vars)

    np.testing.assert_allclose(std_hedging_cost/1e6, 922, rtol=0.03)
    np.testing.assert_allclose(gto_cost/1e6, 902, rtol=0.03)

    si_object.get_allowed_idling_directions(init_state)
    assert si_object._current_regime == "gto"
    assert si_object._original_target_dyn_bot_set == set([0])

    next_state = np.array([2500, 0, 4000])[:, np.newaxis]
    next_w = workload_mat @ next_state
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(next_w,
                                                                                        cw_vars)
    assert not si_object._has_gto_regime_changed(cw_vars)
    si_object.get_allowed_idling_directions(next_state)
    assert si_object._current_regime == "gto"
    assert si_object._original_target_dyn_bot_set == set([0])

    next_state = np.array([0, 0, 10000])[:, np.newaxis]
    next_w = workload_mat @ next_state
    cw_vars = super(StrategicIdlingForesight, si_object)._handle_switching_curve_regime(next_w,
                                                                                        cw_vars)
    assert si_object._has_gto_regime_changed(cw_vars)
    si_object.get_allowed_idling_directions(next_state)
    assert si_object._current_regime == "standard_hedging"
    assert si_object._original_target_dyn_bot_set == set([0])
