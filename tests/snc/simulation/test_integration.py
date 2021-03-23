import pytest
import numpy as np

from snc.agents.hedgehog.hh_agents.big_step_hedgehog_agent import BigStepHedgehogAgent
from snc.agents.hedgehog.hh_agents.big_step_hedgehog_gto_agent import BigStepHedgehogGTOAgent
import snc.environments.scenarios as scenarios
import snc.simulation.snc_simulator as ps
import snc.simulation.utils.load_agents as load_agents


SIM_STEPS = 300
SEED_NO = 42


SKIPPED_TESTS = [
    'demand_node',
    'double_demand_node',
]

PULL_MODELS = [
    'complex_demand_driven_model',
    'complex_demand_driven_model_hot_lots',
    'double_reentrant_line_with_demand_model',
    'double_reentrant_line_with_demand_only_shared_resources_model',
    'multiple_demand_model',
    'one_warehouse',
    'product_demo_beer_kegs',
    'single_station_demand_model',
    'simple_reentrant_line_with_demand_model',
    'tandem_demand_model',
    'three_warehouses_simplified',
    'two_warehouses_simplified',
    'two_warehouses',
    'willems_example_2',
]

PUSH_PULL_MODELS = [
    'push_pull',
    'push_pull_minimal'
]

# Stationary MPC policy cannot handle one activity shared by multiple resources, i.e. not orthogonal
# columns of constituency matrix.
MIP_REQUIRED_MODELS = [
    'input_queued_switch_3x3_model'
]


@pytest.mark.parametrize("scenario_name", sorted(scenarios.SCENARIO_CONSTRUCTORS))
@pytest.mark.parametrize("agent_class", [BigStepHedgehogAgent, BigStepHedgehogGTOAgent])
def test_scenario(scenario_name, agent_class):
    """ Run a brief integration test on a given scenario
    """
    skip_tests = SKIPPED_TESTS + PULL_MODELS + PUSH_PULL_MODELS
    if scenario_name in skip_tests:
        pytest.skip()

    np.random.seed(SEED_NO)

    _, env = scenarios.load_scenario(scenario_name, SEED_NO)
    # Update parameters for quick tests.
    overrides = {
        "HedgehogHyperParams": {
            "theta_0": 0.5,
            "horizon_drain_time_ratio": 0.1,
            "horizon_mpc_ratio": 0.1,
            "minimum_horizon": 10
        },
        "AsymptoticCovarianceParams": {
            "num_presimulation_steps": 100,
            "num_batch": 20
        }
    }

    if scenario_name in MIP_REQUIRED_MODELS:
        overrides["HedgehogHyperParams"]["mpc_policy_class_name"] = "FeedbackMipFeasibleMpcPolicy"

    ac_params, wk_params, si_params, po_params, hh_params, si_class, dp_params, name \
        = load_agents.get_hedgehog_hyperparams(**overrides)
    discount_factor = 0.95
    if agent_class == BigStepHedgehogAgent:
        agent = agent_class(env, discount_factor, wk_params, hh_params, ac_params,
                            si_params, po_params, si_class, dp_params, name)
    elif agent_class == BigStepHedgehogGTOAgent:
        agent = agent_class(env, discount_factor, wk_params, hh_params, ac_params,
                            po_params, dp_params, name)
    else:
        assert False, f"Not recognised agent: {agent_class}"
    simulator = ps.SncSimulator(env, agent, discount_factor=discount_factor)
    simulator.run(num_simulation_steps=SIM_STEPS)
