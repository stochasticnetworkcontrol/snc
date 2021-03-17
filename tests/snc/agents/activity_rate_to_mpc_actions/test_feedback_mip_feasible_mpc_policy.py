import numpy as np
from snc.agents.activity_rate_to_mpc_actions.feedback_mip_feasible_mpc_policy \
    import FeedbackMipFeasibleMpcPolicy


def get_mpc_policy_sirl():
    # Simple reentrant line like environment.
    constituency_matrix = np.array([[1, 0, 1], [0, 1, 0]])
    buffer_processing_matrix = np.array([[-1, 0, 0], [1, -1, 0], [0, 1, -1]])
    return FeedbackMipFeasibleMpcPolicy(constituency_matrix, buffer_processing_matrix)


def get_mpc_policy_routing():
    constituency_matrix = np.eye(3)
    buffer_processing_matrix = np.array([[-1, -1, -1]])
    return FeedbackMipFeasibleMpcPolicy(constituency_matrix, buffer_processing_matrix)


def get_mpc_policy_simple_link_routing_from_book():
    mu12 = 1
    mu13 = 1
    mu25 = 1
    mu32 = 1
    mu34 = 1
    mu35 = 1
    mu45 = 1
    mu5 = 1
    buffer_processing_matrix = np.array([[-mu12, -mu13, 0, 0, 0, 0, 0, 0],
                                         [mu12, 0, -mu25, mu32, 0, 0, 0, 0],
                                         [0, mu13, 0, -mu32, -mu34, -mu35, 0, 0],
                                         [0, 0, 0, 0, mu34, 0, -mu45, 0],
                                         [0, 0, mu25, 0, 0, mu35, mu45, -mu5]])
    constituency_matrix = np.eye(8)
    return FeedbackMipFeasibleMpcPolicy(constituency_matrix, buffer_processing_matrix)


def test_get_nonidling_resources_zero_action():
    sum_actions = np.array([[0], [0], [0]])
    mpc_policy = get_mpc_policy_sirl()
    nonidling_constituency_mat, nonidling_ones = mpc_policy.get_nonidling_resources(sum_actions)
    assert np.all(nonidling_constituency_mat == np.zeros((2, 3)))
    assert np.all(nonidling_ones == np.zeros((2, 1)))


def test_get_nonidling_resources_zero_action_res_1():
    sum_actions = np.array([[0], [1], [0]])
    mpc_policy = get_mpc_policy_sirl()
    nonidling_constituency_mat, nonidling_ones = mpc_policy.get_nonidling_resources(sum_actions)
    assert np.all(nonidling_constituency_mat == np.array([[0, 0, 0], [0, 1, 0]]))
    assert np.all(nonidling_ones == np.array([[0], [1]]))


def test_get_nonidling_resources_zero_action_res_2():
    sum_actions = np.array([[1], [0], [0]])
    mpc_policy = get_mpc_policy_sirl()
    nonidling_constituency_mat, nonidling_ones = mpc_policy.get_nonidling_resources(sum_actions)
    assert np.all(nonidling_constituency_mat == np.array([[1, 0, 1], [0, 0, 0]]))
    assert np.all(nonidling_ones == np.array([[1], [0]]))


def test_get_nonidling_resources_both_active():
    sum_actions = np.array([[0], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    nonidling_constituency_mat, nonidling_ones = mpc_policy.get_nonidling_resources(sum_actions)
    assert np.all(nonidling_constituency_mat == np.array([[1, 0, 1], [0, 1, 0]]))
    assert np.all(nonidling_ones == np.ones((2, 1)))


def test_generate_actions_with_feedback_empty_buffers():
    sum_actions = np.ones((3, 1))
    state = np.zeros((3, 1))
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.zeros((3, 1)))


def test_generate_actions_with_feedback_empty_buffer_1():
    sum_actions = np.ones((3, 1))
    state = np.array([[0], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[0], [1], [1]]))


def test_generate_actions_with_feedback_empty_buffer_1_no_action_buffer_2():
    sum_actions = np.array([[1], [1], [0]])
    state = np.array([[0], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[0], [1], [1]]))


def test_generate_actions_with_feedback_empty_buffers_1_and_3():
    sum_actions = np.array([[0], [1], [0]])
    state = np.array([[0], [1], [0]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[0], [1], [0]]))


def test_generate_actions_with_feedback_priority_buffer_1():
    sum_actions = np.array([[1001], [1000], [1000]])
    state = np.array([[1], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[1], [1], [0]]))


def test_generate_actions_with_feedback_priority_buffer_3():
    sum_actions = np.array([[1000], [1000], [1001]])
    state = np.array([[1], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[0], [1], [1]]))


def test_generate_actions_with_feedback_no_priority():
    sum_actions = np.array([[1000], [1000], [1000]])
    state = np.array([[1], [1], [1]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert action[1] == 1
    assert action[0] == 1 or action[2] == 1


def test_generate_actions_with_feedback_priority_buffer_3_but_empty():
    sum_actions = np.array([[1000], [1000], [1001]])
    state = np.array([[1], [1], [0]])
    mpc_policy = get_mpc_policy_sirl()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.array([[1], [1], [0]]))


def test_generate_actions_with_feedback_routing_enough_items():
    sum_actions = np.array([[1], [1], [1]])
    state = np.array([[3]])
    mpc_policy = get_mpc_policy_routing()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.all(action == np.ones((3, 1)))


def test_generate_actions_with_feedback_routing_only_one_item():
    sum_actions = np.array([[1], [1], [1]])
    state = np.array([[1]])
    mpc_policy = get_mpc_policy_routing()
    action = mpc_policy.generate_actions_with_feedback(sum_actions, state)
    assert np.sum(action) == 1


def test_get_actions_drain_each_buffer_routing():
    mpc_policy = get_mpc_policy_routing()
    actions_drain_each_buffer = mpc_policy.get_actions_drain_each_buffer()
    assert np.all(actions_drain_each_buffer[0] == [np.array([0, 1, 2])])


def test_get_action_drain_each_buffer_simple_link_routing():
    mpc_policy = get_mpc_policy_simple_link_routing_from_book()
    actions_drain_each_buffer = mpc_policy.get_actions_drain_each_buffer()
    assert np.all(actions_drain_each_buffer[0] == [np.array([0, 1])])
    assert np.all(actions_drain_each_buffer[1] == [np.array([2])])
    assert np.all(actions_drain_each_buffer[2] == [np.array([3, 4, 5])])
    assert np.all(actions_drain_each_buffer[3] == [np.array([6])])
    assert np.all(actions_drain_each_buffer[4] == [np.array([7])])


def test_update_bias_counter_routing_enough_items():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[3]])
    action = np.array([[1], [1], [1]])
    sum_actions = np.ones((3, 1))
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.zeros((3, 1)))


def test_update_bias_counter_routing_enough_items_not_required():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[3]])
    action = np.array([[1], [1], [1]])
    sum_actions = np.zeros((3, 1))
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.zeros((3, 1)))


def test_update_bias_counter_routing_not_enough_items_1():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[2]])
    action = np.array([[1], [1], [0]])
    sum_actions = np.ones((3, 1))
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.array([[0], [0], [1]]))


def test_update_bias_counter_routing_not_enough_items_1_not_required():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[2]])
    action = np.array([[1], [1], [0]])
    sum_actions = np.array([[1], [1], [0]])
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.zeros((3, 1)))


def test_update_bias_counter_routing_not_enough_items_1_other_action():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[2]])
    action = np.array([[0], [1], [1]])
    sum_actions = np.ones((3, 1))
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.array([[1], [0], [0]]))


def test_update_bias_counter_routing_not_enough_items_2():
    mpc_policy = get_mpc_policy_routing()
    state = np.array([[1]])
    action = np.array([[0], [1], [0]])
    sum_actions = np.ones((3, 1))
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.array([[1], [0], [1]]))


def test_update_bias_counter_simple_link_routing():
    mpc_policy = get_mpc_policy_simple_link_routing_from_book()
    state = np.ones((5, 1))
    action = np.array([1, 0, 1, 1, 0, 0, 1, 1])[:, None]
    sum_actions = np.ones_like(action)
    mpc_policy.update_bias_counter(state, action, sum_actions)
    assert np.all(mpc_policy._bias_counter.value == np.array([0, 1, 0, 0, 1, 1, 0, 0])[:, None])

