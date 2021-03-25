import pytest

from src.snc import PureFeedbackMIPHedgehogAgent
from src.snc.environments import examples
from src.snc.simulation.utils import get_hedgehog_hyperparams


def test_create_agent_with_mip_flag_set_to_false():
    env = examples.double_reentrant_line_model()
    discount_factor = 0.9999
    overrides = {
        'BigStepPenaltyPolicyParams': {
            'boolean_action_flag': False,  # This should throw an error.
            'convex_solver': 'cvx.CPLEX'
        },
        'HedgehogHyperParams': {'activity_rates_policy_class_name': 'BigStepPolicy'}
    }
    ac_params, wk_params, si_params, po_params, hh_params, _, dp_params, _ \
        = get_hedgehog_hyperparams(**overrides)
    with pytest.raises(AssertionError):
        _ = PureFeedbackMIPHedgehogAgent(
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
    overrides = {
        'BigStepPenaltyPolicyParams': {
            'boolean_action_flag': True,  # This should throw an error.
            'convex_solver': 'cvx.CPLEX'
        },
        'HedgehogHyperParams': {
            'activity_rates_policy_class_name': 'BigStepPolicy',
            'horizon_mpc_ratio': 0,
            'minimum_horizon': 1
        }
    }
    ac_params, wk_params, si_params, po_params, hh_params, _, dp_params, _ \
        = get_hedgehog_hyperparams(**overrides)
    pf_mip_agent = PureFeedbackMIPHedgehogAgent(
        env,
        discount_factor,
        wk_params,
        hh_params,
        ac_params,
        si_params,
        po_params,
        demand_planning_params=dp_params
    )
    assert pf_mip_agent.get_horizon() == 1
