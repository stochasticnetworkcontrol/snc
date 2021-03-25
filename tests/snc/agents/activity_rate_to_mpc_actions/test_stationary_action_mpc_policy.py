import numpy as np
from collections import namedtuple
from snc.agents.activity_rate_to_mpc_actions.stationary_mpc_policy \
    import StationaryActionMPCPolicy
from snc.utils.snc_tools import is_binary
import pytest

mock_action_mpc_policy = namedtuple('MockMPC', 'np_random')


def test_init_stationary_action_mpc_policy_from_constituency_matrix_with_multiple_ones_per_column():
    constituency_matrix = np.array([[1, 0], [1, 1]])
    with pytest.raises(AssertionError):
        _ = StationaryActionMPCPolicy(constituency_matrix)


def test_build_stationary_policy_matrix_from_negative_z_star():
    z_star = np.array([[-0.3], [0.7]])
    mpc_policy = StationaryActionMPCPolicy(np.eye(2))
    with pytest.raises(AssertionError):
        _ = mpc_policy.build_stationary_policy_matrix(z_star)


def test_build_stationary_policy_matrix_from_z_star_with_too_large_components():
    z_star = np.array([[1.1], [0]])
    mpc_policy = StationaryActionMPCPolicy(np.eye(2))
    with pytest.raises(AssertionError):
        _ = mpc_policy.build_stationary_policy_matrix(z_star)


def test_build_stationary_policy_matrix_for_1_resource_2_activities():
    constituency_matrix = np.array([[1, 1]])
    z_star = np.array([[0.3], [0.7]])
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    policy_matrix = mpc_policy.build_stationary_policy_matrix(z_star)
    policy_matrix_theory = np.vstack((z_star, 0)).T
    assert np.all(policy_matrix == policy_matrix_theory)


def test_build_stationary_policy_matrix_for_1_resource_2_activities_positvie_idling_prob():
    constituency_matrix = np.array([[1, 1]])
    z_star = np.array([[0.3], [0.1]])
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    policy_matrix = mpc_policy.build_stationary_policy_matrix(z_star)
    policy_matrix_theory = np.vstack((z_star, 0.6)).T
    assert np.all(policy_matrix == policy_matrix_theory)


def test_build_stationary_policy_matrix_for_2_resources_activities():
    constituency_matrix = np.eye(2)
    z_star = np.array([[0.3], [0.7]])
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    policy_matrix = mpc_policy.build_stationary_policy_matrix(z_star)
    policy_matrix_theory = np.array([[0.3, 0, 0.7], [0, 0.7, 0.3]])
    np.testing.assert_almost_equal(policy_matrix, policy_matrix_theory)


def test_build_stationary_policy_matrix_for_many_resources_and_activities_positive_idling_prob():
    num_resources = 1000
    constituency_matrix = np.eye(num_resources)
    z_star = np.random.dirichlet(0.5 * np.ones((num_resources,)))
    np.testing.assert_almost_equal(np.sum(z_star), 1)
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    policy_matrix = mpc_policy.build_stationary_policy_matrix(z_star)

    idle = 1 - z_star.reshape((num_resources, 1))
    policy_matrix_theory = np.hstack((np.diag(z_star), idle))
    np.testing.assert_almost_equal(policy_matrix, policy_matrix_theory)


def test_draw_samples_from_stationary_policy_matrix_all_zeros():
    num_mpc_steps = 10
    num_resources = 10
    num_activities = 20
    policy_matrix = np.hstack(
        (np.zeros((num_resources, num_activities)), np.ones((num_resources, 1))))
    actions = StationaryActionMPCPolicy.draw_samples_from_stationary_policy_matrix(
        policy_matrix, num_mpc_steps)
    assert is_binary(actions)
    assert np.all(actions == np.zeros((num_activities, num_mpc_steps)))


def test_draw_samples_from_stationary_policy_matrix_all_actions_with_prob_one():
    np.random.seed(42)
    num_mpc_steps = int(1e3)
    num_resources = 4
    policy_matrix = np.hstack((np.eye(num_resources), np.zeros((num_resources, 1))))
    actions = StationaryActionMPCPolicy.draw_samples_from_stationary_policy_matrix(
        policy_matrix, num_mpc_steps)
    assert is_binary(actions)
    assert np.all(actions == np.ones((1, num_mpc_steps)))


def test_draw_samples_from_stationary_policy_matrix_some_distribution_3_resources_4_actions():
    np.random.seed(42)
    num_mpc_steps = int(1e5)
    policy_matrix = np.array([[0.45, 0.55, 0, 0, 0], [0, 0, 0.3, 0, 0.7], [0, 0, 0, 1, 0]])
    actions = StationaryActionMPCPolicy.draw_samples_from_stationary_policy_matrix(
        policy_matrix, num_mpc_steps)
    assert is_binary(actions)
    actions_freq = np.sum(actions, axis=1) / num_mpc_steps
    actions_freq_theory = np.sum(policy_matrix[:, 0:-1], axis=0)
    np.testing.assert_almost_equal(actions_freq, actions_freq_theory, decimal=2)


def test_mpc_action_many_random_distributions():
    np.random.seed(42)
    num_tests = 20
    num_mpc_steps = int(1e4)
    num_resources = 10
    num_activities_per_resource = 3
    # Build random probability distributions with extra idling activity
    policy_matrix = np.zeros((num_resources, num_resources * num_activities_per_resource + 1))
    for i in range(num_resources):
        p = np.random.dirichlet(0.5 * np.ones((num_activities_per_resource + 1,)), size=1)
        policy_matrix[i, i * num_activities_per_resource:(i + 1) * num_activities_per_resource] = \
            p[0, :-1]
        policy_matrix[i, -1] = p[0, -1]

    for i in range(num_tests):
        actions = StationaryActionMPCPolicy.draw_samples_from_stationary_policy_matrix(
            policy_matrix, num_mpc_steps)
        assert is_binary(actions)
        actions_freq = np.sum(actions, axis=1) / num_mpc_steps
        actions_freq_theory = np.sum(policy_matrix[:, 0:-1], axis=0)
        np.testing.assert_almost_equal(actions_freq, actions_freq_theory, decimal=2)


def test_rounded_stationary_mpc_action_zeros():
    num_mpc_steps = 10
    num_activities = 3
    z_star = np.zeros((num_activities, 1))
    constituency_matrix = np.eye(num_activities)
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(actions == np.zeros((num_activities, num_mpc_steps)))


def test_rounded_stationary_mpc_action_all_near_zero_independent_resource():
    num_mpc_steps = 10
    num_activities = 3
    z_star = 1e-7 * (np.random.random_sample((num_activities, 1)) - 1)
    constituency_matrix = np.eye(num_activities)
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    np.testing.assert_almost_equal(actions, np.zeros((num_activities, num_mpc_steps)))


def test_rounded_stationary_mpc_action_all_near_zero_shared_resource():
    np.random.seed(42)
    num_mpc_steps = 10
    num_activities = 3
    z_star = 1e-7 * (np.random.random_sample((num_activities, 1)) - 1)
    constituency_matrix = np.ones((1, num_activities))
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    np.testing.assert_almost_equal(actions, np.zeros((num_activities, num_mpc_steps)))


def test_rounded_mpc_action_probability_one():
    np.random.seed(42)
    num_mpc_steps = int(1e3)
    z_star = np.array([1, 0])[:, None]
    constituency_matrix = np.ones((1, 2))
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(actions == np.vstack((np.ones((1, num_mpc_steps)),
                                        np.zeros((1, num_mpc_steps)))))


def test_rounded_mpc_action_some_distribution_one_resource():
    np.random.seed(42)
    num_mpc_steps = int(1e5)
    z_star = np.array([0.45, 0.55])[:, None]
    constituency_matrix = np.ones((1, 2))
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(constituency_matrix @ actions == np.ones((1, num_mpc_steps)))
    actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
    np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_mpc_action_many_random_distributions_2_resource():
    np.random.seed(42)
    num_mpc_steps = int(1e5)
    num_tests = 50
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1]])
    for i in range(num_tests):
        p = np.random.random_sample((3, 1))
        z_star = p / np.sum(p)
        mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
        actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
        assert is_binary(actions)
        assert np.all(constituency_matrix @ actions <= np.ones((1, num_mpc_steps)))
        actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
        np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_rounded_mpc_action_some_distribution_one_resource_with_positive_prob_of_idling():
    np.random.seed(42)
    num_mpc_steps = int(1e4)
    z_star = np.array([0.2, 0.3])[:, None]
    constituency_matrix = np.ones((1, 2))
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(np.sum(actions, axis=0) <= np.ones((1, num_mpc_steps)))
    actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
    np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_rounded_mpc_action_some_distribution_two_partially_shared_resources():
    np.random.seed(42)
    num_mpc_steps = int(1e5)
    z_star = np.array([0.8, 0.2, 1])[:, None]
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1]])
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(constituency_matrix @ actions == np.ones((2, num_mpc_steps)))
    actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
    np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_rounded_mpc_action_some_distribution_two_partially_shared_resources_with_idling():
    np.random.seed(42)
    num_mpc_steps = int(1e4)
    z_star = np.array([0.1, 0.2, 0.3])[:, None]
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1]])
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)
    assert is_binary(actions)
    assert np.all(constituency_matrix @ actions <= np.ones((2, num_mpc_steps)))
    actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
    np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_rounded_mpc_action_multiple_resources_owning_same_activity_but_passed_correctly():
    np.random.seed(42)
    num_mpc_steps = int(1e4)
    z_star = np.array([0.1, 0.2, 0.3])[:, None]
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]])
    physical_constituency_matrix = constituency_matrix[0:-1, :]

    mpc_policy = StationaryActionMPCPolicy(physical_constituency_matrix)
    actions = mpc_policy.obtain_actions(num_mpc_steps=num_mpc_steps, z_star=z_star)

    assert is_binary(actions)
    assert np.all(constituency_matrix[0:-1, :] @ actions <= np.ones((2, num_mpc_steps)))
    actions_freq = (np.sum(actions, axis=1) / num_mpc_steps)[:, None]
    np.testing.assert_almost_equal(actions_freq, z_star, decimal=2)


def test_obtain_actions_no_num_mpc_steps():
    num_activities = 3
    z_star = np.zeros((num_activities, 1))
    constituency_matrix = np.eye(num_activities)
    mpc_policy = StationaryActionMPCPolicy(constituency_matrix)
    actions = mpc_policy.obtain_actions(z_star=z_star)
    assert np.all(actions == np.zeros((num_activities, 1)))
