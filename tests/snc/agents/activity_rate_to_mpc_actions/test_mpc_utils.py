import pytest
import src.snc.agents.activity_rate_to_mpc_actions.mpc_utils as mpc_utils


def test_check_num_time_steps_float():
    num_time_steps = 1.2
    with pytest.raises(AssertionError):
        mpc_utils.check_num_time_steps(num_time_steps)


def test_check_num_time_steps_null():
    num_time_steps = 0
    with pytest.raises(AssertionError):
        mpc_utils.check_num_time_steps(num_time_steps)


def test_check_num_time_steps_negative():
    num_time_steps = -2
    with pytest.raises(AssertionError):
        mpc_utils.check_num_time_steps(num_time_steps)

