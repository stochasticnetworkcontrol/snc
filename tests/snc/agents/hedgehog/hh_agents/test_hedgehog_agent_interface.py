from snc.agents.hedgehog.hh_agents.hedgehog_agent_interface import HedgehogAgentInterface


def test_get_num_steps_to_recompute_policy_not_integer_horizon():
    current_horizon = 30.8
    horizon_mpc_ratio = 0.1
    minimum_num_steps = 1
    num_steps = HedgehogAgentInterface.get_num_steps_to_recompute_policy(
        current_horizon, horizon_mpc_ratio, minimum_num_steps)
    assert num_steps == 4


def test_get_num_steps_to_recompute_policy_zero_mpc_ration():
    current_horizon = 30.8
    horizon_mpc_ratio = 0
    minimum_num_steps = 1
    num_steps = HedgehogAgentInterface.get_num_steps_to_recompute_policy(
        current_horizon, horizon_mpc_ratio, minimum_num_steps)
    assert num_steps == 1

