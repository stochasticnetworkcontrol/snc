from typing import cast

from src.snc import \
    AsymptoticCovarianceParams, \
    BigStepLayeredPolicyParams, \
    BigStepPenaltyPolicyParams, \
    DemandPlanningParams, \
    HedgehogHyperParams, \
    StrategicIdlingParams, \
    WorkloadRelaxationParams
import src.snc.simulation.utils.load_agents as load_agents


def get_hedgehog_default_values(activity_rates_policy_class_name="BigStepLayeredPolicy"):
    ac_params = AsymptoticCovarianceParams(
        num_presimulation_steps=10000, num_batch=200)
    wk_params = WorkloadRelaxationParams(
        num_vectors=None, load_threshold=None, convex_solver='cvx.CPLEX')
    si_params = StrategicIdlingParams(
        strategic_idling_class='StrategicIdlingForesight',
        convex_solver='cvx.CPLEX',
        epsilon=0.05,
        shift_eps=1e-2,
        hedging_scaling_factor=1,
        penalty_coeff_w_star=1e-5)
    si_class = load_agents.get_strategic_idling_class(
        cast(StrategicIdlingParams, si_params).strategic_idling_class)
    if activity_rates_policy_class_name == "BigStepLayeredPolicy":
        po_params = BigStepLayeredPolicyParams(convex_solver='cvx.CPLEX')
    elif activity_rates_policy_class_name == "BigStepPenaltyPolicy":
        po_params = BigStepPenaltyPolicyParams(
            convex_solver='cvx.CPLEX',
            boolean_action_flag=False,
            safety_penalty_coeff=10,
            nonidling_penalty_coeff=1000
        )
    hh_params = HedgehogHyperParams(
        activity_rates_policy_class_name=activity_rates_policy_class_name,
        horizon_drain_time_ratio=0,
        horizon_mpc_ratio=1,
        minimum_horizon=100,
        mpc_policy_class_name="FeedbackStationaryFeasibleMpcPolicy",
        theta_0=0.5
    )
    dp_params = DemandPlanningParams(
        demand_planning_class_name=None,
        params_dict=dict()
    )
    name = "BigStepHedgehogAgent"
    return ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name


def assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name):
    assert updated_params[0] == ac_params
    assert updated_params[1] == wk_params
    assert updated_params[2] == si_params
    assert updated_params[3] == po_params
    assert updated_params[4] == hh_params
    assert updated_params[5] == si_class
    assert updated_params[6] == dp_params
    assert updated_params[7] == name


def get_maxweight_default_params():
    weight_per_buffer = None
    name = "MaxWeightAgent"
    return weight_per_buffer, name


def assert_equal_maxweight_parameters(updated_params, weight_per_buffer, name):
    assert updated_params[0] == weight_per_buffer
    assert updated_params[1] == name


def test_get_hedgehog_hyperparams_default_values():
    default_params = load_agents.get_hedgehog_hyperparams()
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_default_values()
    assert_equal_hedgehog_parameters(default_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_asymptotic_cov():
    overrides = {'AsymptoticCovarianceParams': {'num_presimulation_steps': 2, 'num_batch': 1}}
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    ac_params = AsymptoticCovarianceParams(**overrides['AsymptoticCovarianceParams'])
    _, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_default_values()
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params,
                                     po_params, hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_workload():
    overrides = {'WorkloadRelaxationParams': {'num_vectors': 2, 'load_threshold': 0.1,
                                              'convex_solver': 'fake_solver'}}
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    wk_params = WorkloadRelaxationParams(**overrides['WorkloadRelaxationParams'])
    ac_params, _, si_params, po_params, hh_params, si_class, dp_params, name \
        = get_hedgehog_default_values()
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_strategic_idling():
    overrides = {'StrategicIdlingParams': {
        'strategic_idling_class': 'StrategicIdlingHedgehogNaiveGTO',
        'convex_solver': 'fake_solver',
        'epsilon': 5e6,
        'shift_eps': 0.33,
        'hedging_scaling_factor': 30,
        'penalty_coeff_w_star': 3}}
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    si_params = StrategicIdlingParams(**overrides['StrategicIdlingParams'])
    si_class = load_agents.get_strategic_idling_class(
        cast(StrategicIdlingParams, si_params).strategic_idling_class)
    ac_params, wk_params, _, po_params, hh_params, _, dp_params, name \
        = get_hedgehog_default_values()
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_big_step_layered_policy():
    overrides = {
        'BigStepLayeredPolicyParams': {
            'convex_solver': 'fake_solver'
        },
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepLayeredPolicy'
        }
    }
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    po_params = BigStepLayeredPolicyParams(**overrides['BigStepLayeredPolicyParams'])
    ac_params, wk_params, si_params, _, hh_params, si_class, dp_params, name \
        = get_hedgehog_default_values('BigStepLayeredPolicy')
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_big_step_penalty_policy():
    overrides = {
        'BigStepPenaltyPolicyParams': {
            'convex_solver': 'fake_solver',
            'boolean_action_flag': True,
            'safety_penalty_coeff': 33,
            'nonidling_penalty_coeff': 55
        },
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepPenaltyPolicy'
        }
    }
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    po_params = BigStepPenaltyPolicyParams(**overrides['BigStepPenaltyPolicyParams'])
    ac_params, wk_params, si_params, _, hh_params, si_class, dp_params, name \
        = get_hedgehog_default_values('BigStepPenaltyPolicy')
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_hedgehog_hyperparams():
    overrides = {
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepPenaltyPolicy',
            'horizon_drain_time_ratio': 0.99,
            'horizon_mpc_ratio': 0.55,
            'minimum_horizon': 0.77,
            'mpc_policy_class_name': 'fake_mpc_class',
            'theta_0': 0.12
        }
    }
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    hh_params = HedgehogHyperParams(**overrides['HedgehogHyperParams'])
    ac_params, wk_params, si_params, po_params, _, si_class, dp_params, name \
        = get_hedgehog_default_values('BigStepPenaltyPolicy')
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_name():
    overrides = {'name': "funny_name"}
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    name = overrides['name']
    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, _ \
        = get_hedgehog_default_values('BigStepLayeredPolicy')
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_multiple():
    overrides = {
        'BigStepPenaltyPolicyParams': {
            'boolean_action_flag': False,
            'convex_solver': 'fake_solver',
            'safety_penalty_coeff': 21,
            'nonidling_penalty_coeff': 12
        },
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepPenaltyPolicy',
            'horizon_drain_time_ratio': 0.99,
            'horizon_mpc_ratio': 0.55,
            'minimum_horizon': 0.77,
            'mpc_policy_class_name': 'fake_mpc_class',
            'theta_0': 99
        },
        'name': 'funny_name'
    }
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    po_params = BigStepPenaltyPolicyParams(**overrides['BigStepPenaltyPolicyParams'])
    hh_params = HedgehogHyperParams(**overrides['HedgehogHyperParams'])
    name = overrides['name']
    ac_params, wk_params, si_params, _, _, si_class, dp_params, _ \
        = get_hedgehog_default_values('BigStepPenaltyPolicy')
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_hedgehog_hyperparams_overwrite_demand_planning_params():
    overrides = {
        'DemandPlanningParams': {
            'demand_planning_class_name': 'fake_class',
            'params_dict': {'fake_key_1': 1, 'fake_key_2': 2}
        }
    }
    updated_params = load_agents.get_hedgehog_hyperparams(**overrides)
    dp_params = DemandPlanningParams(**overrides['DemandPlanningParams'])
    ac_params, wk_params, si_params, po_params, hh_params, si_class, _, name \
        = get_hedgehog_default_values()
    assert_equal_hedgehog_parameters(updated_params, ac_params, wk_params, si_params, po_params,
                                     hh_params, si_class, dp_params, name)


def test_get_maxweight_parameters():
    default_params = load_agents.get_maxweight_params()
    weight_per_buffer, name = get_maxweight_default_params()
    assert_equal_maxweight_parameters(default_params, weight_per_buffer, name)


def test_get_maxweight_parameters_overwrite_weight_per_buffer():
    weight_per_buffer = [[3], [4], [5]]
    overrides = {'weight_per_buffer': weight_per_buffer}
    updated_params = load_agents.get_maxweight_params(**overrides)
    _, name = get_maxweight_default_params()
    assert_equal_maxweight_parameters(updated_params, weight_per_buffer, name)


def test_get_maxweight_parameters_overwrite_name():
    name = "funny_name"
    overrides = {'name': name}
    updated_params = load_agents.get_maxweight_params(**overrides)
    weight_per_buffer, _ = get_maxweight_default_params()
    assert_equal_maxweight_parameters(updated_params, weight_per_buffer, name)


def test_get_maxweight_parameters_overwrite_multiple():
    weight_per_buffer = [[3], [4], [5]]
    name = "funny_name"
    overrides = {'weight_per_buffer': weight_per_buffer, 'name': name}
    updated_params = load_agents.get_maxweight_params(**overrides)
    assert_equal_maxweight_parameters(updated_params, weight_per_buffer, name)
