import numpy as np
import pytest
from src.snc \
    import NoMPCPolicy


def test_no_binary_activity_rates():
    z_star = np.array([[0.1], [0.8]])
    constituency_matrix = np.eye(2)
    mpc_policy = NoMPCPolicy(constituency_matrix)
    with pytest.raises(AssertionError):
        mpc_policy.obtain_actions(z_star=z_star)


def test_approx_binary_activity_rates():
    z_star = np.array([[1-1e-7], [0+1e-7]])
    constituency_matrix = np.eye(2)
    mpc_policy = NoMPCPolicy(constituency_matrix)
    assert np.all(z_star == mpc_policy.obtain_actions(z_star=z_star))


def test_binary_activity_rates():
    z_star = np.array([[1], [0]])
    constituency_matrix = np.eye(2)
    mpc_policy = NoMPCPolicy(constituency_matrix)
    assert np.all(z_star == mpc_policy.obtain_actions(z_star=z_star))


def test_violate_constituency_matrix_constraints():
    z_star = np.array([[1], [1]])
    constituency_matrix = np.array([[1, 1]])
    mpc_policy = NoMPCPolicy(constituency_matrix)
    with pytest.raises(AssertionError):
        mpc_policy.obtain_actions(z_star=z_star)
