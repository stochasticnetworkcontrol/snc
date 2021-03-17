import pytest

from snc.agents.hedgehog.hh_agents.\
    pure_feedback_stationary_hedgehog_agent import PureFeedbackStationaryHedgehogAgent
from snc.environments import examples
from snc.simulation.utils.load_agents import get_hedgehog_hyperparams


def test_create_agent_with_mip_flag_set_to_true():
    env = examples.double_reentrant_line_model()
    discount_factor = 0.9999
    overrides = {
        'BigStepPenaltyPolicyParams': {
            'boolean_action_flag': True,  # This should throw an error.
            'convex_solver': 'cvx.CPLEX'
        },
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepPenaltyPolicy'
        }
    }
    ac_params, wk_params, si_params, po_params, hh_params, _, _, dp_params \
        = get_hedgehog_hyperparams(**overrides)
    with pytest.raises(AssertionError):
        _ = PureFeedbackStationaryHedgehogAgent(
            env,
            discount_factor,
            wk_params,
            hh_params,
            ac_params,
            si_params,
            po_params,
            demand_planning_params=dp_params
        )


def test_get_horizon():
    env = examples.double_reentrant_line_model()
    discount_factor = 0.9999
    overrides = {'HedgehogHyperParams': {'horizon_mpc_ratio': 0, 'minimum_horizon': 1}}
    ac_params, wk_params, si_params, _, hh_params, _, _, dp_params \
        = get_hedgehog_hyperparams(**overrides)
    pf_mip_agent = PureFeedbackStationaryHedgehogAgent(
        env,
        discount_factor,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        demand_planning_params=dp_params
    )
    assert pf_mip_agent.get_horizon() == 1
