import os
from argparse import Namespace

import pytest
from mock import MagicMock, patch

from src import snc
import src.snc.simulation.validation_script as validation_script


def get_args(agents,
             hedgehog_param_overrides="{'AsymptoticCovarianceParams': {'num_batch': 10, "
                                      "'num_presimulation_steps': 100}}"):
    args = Namespace(
        agents=agents,
        art_run=False,
        debug_info=False,
        discount_factor=0.999999,
        env_name='simple_reentrant_line',
        env_param_overrides='{}',
        hedgehog_param_overrides=hedgehog_param_overrides,
        maxweight_param_overrides='{}',
        server_mode=True,
        logdir=os.path.join(os.path.dirname(snc.__file__), 'logs'),
        num_steps=200,
        rl_checkpoints=None,
        rl_agent_params=None,
        seed=42)
    return args


def test_validation_script_run_validation_bs_hedgehog():
    """Test run validation, including saving files, for Big Step Hedgehog agent."""
    args = get_args('bs_hedgehog')
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def test_validation_script_run_validation_bs_hedgehog_mip_mpc():
    """Test run validation, including saving files, for Big Step Hedgehog agent."""
    hedgehog_param_overrides \
        = "{'HedgehogHyperParams': {'mpc_policy_class_name': 'FeedbackMipFeasibleMpcPolicy'}}"
    args = get_args('bs_hedgehog', hedgehog_param_overrides)
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def test_validation_script_run_validation_pf_stationary_hedgehog_empty_params_overrides():
    """Test run validation, including saving files, for Pure Feedback with MIP Hedgehog agent."""
    hedgehog_param_overrides \
        = "{'AsymptoticCovarianceParams': {'num_batch': 10, 'num_presimulation_steps': 100}," \
          "'StrategicIdlingParams': {'strategic_idling_class': 'StrategicIdlingHedging'}," \
          "'HedgehogHyperParams': {'activity_rates_policy_class_name': 'BigStepPolicy'," \
          "'horizon_mpc_ratio': 0,'minimum_horizon': 1}}"
    args = get_args("pf_stationary_hedgehog", hedgehog_param_overrides)
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def test_validation_script_run_validation_pf_mip_hedgehog_empty_params_overrides():
    """Test run validation, including saving files, for Pure Feedback with MIP Hedgehog agent."""
    hedgehog_param_overrides \
        = "{'AsymptoticCovarianceParams': {'num_batch': 10, 'num_presimulation_steps': 100}," \
        "'BigStepPenaltyPolicyParams': {'convex_solver': 'cvx.CPLEX', " \
          "'boolean_action_flag': 'True'}," \
        "'StrategicIdlingParams': {'strategic_idling_class': 'StrategicIdlingHedging'}," \
        "'HedgehogHyperParams': {'activity_rates_policy_class_name': 'BigStepPolicy', " \
          "'horizon_mpc_ratio': 0, 'minimum_horizon': 1}}"
    args = get_args("pf_mip_hedgehog", hedgehog_param_overrides)
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def test_validation_script_run_validation_pf_hedgehog():
    """Test run validation, including saving files, for Pure Feedback with MIP Hedgehog agent."""
    hedgehog_param_overrides \
        = "{'BigStepPenaltyPolicyParams': {'boolean_action_flag': True}," \
          "'HedgehogHyperParams': {'activity_rates_policy_class_name': 'BigStepPolicy'," \
          "'horizon_drain_time_ratio': 0, 'horizon_mpc_ratio': 0, 'minimum_horizon': 1}," \
          "'AsymptoticCovarianceParams': {'num_batch': 10, 'num_presimulation_steps': 100}," \
          "'StrategicIdlingParams': {'strategic_idling_class': 'StrategicIdlingHedging'}}"
    args = get_args('pf_mip_hedgehog', hedgehog_param_overrides)
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def test_validation_script_run_validation_pf_stationary_hedgehog():
    """Test run validation, including saving files, for Pure Feedback with stationary MPC Hedgehog
    agent."""
    hedgehog_param_overrides \
        = "{'BigStepPenaltyPolicyParams': {'boolean_action_flag': False}," \
          "'HedgehogHyperParams': {'activity_rates_policy_class_name': 'BigStepPolicy'," \
          "'horizon_drain_time_ratio': 0, 'horizon_mpc_ratio': 0, 'minimum_horizon': 1}," \
          "'AsymptoticCovarianceParams': {'num_batch': 10, 'num_presimulation_steps': 100}," \
          "'StrategicIdlingParams': {'strategic_idling_class': 'StrategicIdlingHedging'}}"
    args = get_args('pf_stationary_hedgehog', hedgehog_param_overrides)
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


@pytest.fixture(params=[
    "{'HedgehogHyperParams': {'horizon_drain_time_ratio': 0.1, "
    "'activity_rates_policy_class_name': 'BigStepPolicy'}}",
    "{'HedgehogHyperParams': {'horizon_mpc_ratio': 0.2,"
    "'activity_rates_policy_class_name': 'BigStepPolicy'}}",
    "{'HedgehogHyperParams': {'minimum_horizon': 3,"
    "'activity_rates_policy_class_name': 'BigStepPolicy'}}"
])
def hh_wrong_overrides_fixture(request):
    return request.param


@pytest.fixture(params=['pf_stationary_hedgehog', 'pf_mip_hedgehog'])
def pf_agent_fixture(request):
    return request.param


def test_validation_script_run_validation_pf_hedgehog_wrong_params(hh_wrong_overrides_fixture,
                                                                   pf_agent_fixture):
    args = get_args(pf_agent_fixture, hh_wrong_overrides_fixture)
    args = validation_script.process_parsed_args(args)
    with pytest.raises(AssertionError):
        validation_script.run_validation(args)


def test_validation_script_run_validation_maxweight():
    """Test run validation, including saving files, for MaxWeight agent."""
    args = get_args('maxweight')
    args = validation_script.process_parsed_args(args)
    validation_script.run_validation(args)


def process_rl_agent_params(args):
    rl_agent_params = validation_script.process_agent_parameters(args.rl_agent_params)
    args.rl_agent_params = validation_script.post_process_rl_agent_params(rl_agent_params,
                                                                          args.agents)
    return args


def test_agent_param_parsing_dict():
    args = Namespace(
        agents=['ppo'],
        rl_agent_params='{"test": 10}'
    )
    args = process_rl_agent_params(args)
    assert args.rl_agent_params == [{"test": 10}]


def test_agent_param_parsing_single_entry_list():
    args = Namespace(
        agents=['ppo'],
        rl_agent_params='[{"test": 10}]'
    )
    args = process_rl_agent_params(args)
    assert args.rl_agent_params == [{"test": 10}]


def test_agent_param_parsing_multi_entry_list():
    args = Namespace(
        agents=['ppo', 'reinforce'],
        rl_agent_params='[{"test": 10}, {"x": [1, 2, 3]}]'
    )
    args = process_rl_agent_params(args)
    assert args.rl_agent_params == [{"test": 10}, {"x": [1, 2, 3]}]


def test_agent_counting_too_few_agents():
    args = Namespace(
        agents=['ppo'],
        rl_agent_params='[{"test": 10}, {"x": [1, 2, 3]}]'
    )
    with pytest.raises(AssertionError, match='The number of agent parameter sets provided'):
        _ = process_rl_agent_params(args)


def test_agent_counting_too_few_param_dicts():
    args = Namespace(
        agents=['ppo', 'ppo'],
        rl_agent_params='[{"test": 10}]'
    )
    with pytest.raises(AssertionError, match='The number of agent parameter sets provided'):
        _ = process_rl_agent_params(args)


def test_open_single_json(mock):
    """Test that passing a single file path leads to a single JSON file being opened."""
    mock_open = mock.mock_open()
    handle = mock_open()
    with patch('snc.simulation.validation_script.open', mock_open):
        import json
        json.load = MagicMock()
        args = Namespace(
            agents=['ppo'],
            rl_agent_params='/'
        )
        _ = process_rl_agent_params(args)
    json.load.assert_called_once_with(handle)


def test_open_multiple_json(mock):
    """Test that passing a multiple file paths leads to a corresponding JSON files being opened."""
    mock_open = mock.mock_open()
    handle = mock_open()
    with patch('snc.simulation.validation_script.open', mock_open):
        import json
        json.load = MagicMock()
        args = Namespace(
            agents=['ppo', 'reinforce'],
            rl_agent_params='["/", "/tmp"]'
        )
        _ = process_rl_agent_params(args)
    assert json.load.call_count == 2
    assert json.load.call_args_list[0][0][0] is handle
    assert json.load.call_args_list[1][0][0] is handle


def test_errors_on_false_path():
    args = Namespace(
        agents=['ppo'],
        rl_agent_params='/no/such/path/exists'
    )
    with pytest.raises(ValueError, match='invalid'):
        _ = process_rl_agent_params(args)
