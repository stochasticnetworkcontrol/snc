import numpy as np
import pytest

import src.snc.environments.job_generators.discrete_review_job_generator \
    as drjg
import src.snc.environments.controlled_random_walk as crw
import src.snc.environments.state_initialiser as si
from src import snc as random_nonidling_agent, snc as custom_priority_agent
import src.snc.agents.general_heuristics.longest_buffer_priority_agent \
    as longest_priority_agent


def get_null_env_params(state, num_resources=None, buffer_processing_matrix=None,
                        constituency_matrix=None):
    num_buffers = state.shape[0]
    arrival_rate = np.ones_like(state)
    if num_resources is None:
        num_resources = num_buffers
    if buffer_processing_matrix is None:
        buffer_processing_matrix = -np.triu(np.ones((num_buffers, num_resources)))
    if constituency_matrix is None:
        constituency_matrix = np.zeros((num_resources, num_resources))
    time_interval = 1

    return {
        "cost_per_buffer": np.zeros_like(state),
        "capacity": np.zeros_like(state),
        "constituency_matrix": constituency_matrix,
        "job_generator": drjg.DeterministicDiscreteReviewJobGenerator(
            arrival_rate, buffer_processing_matrix, sim_time_interval=time_interval
        ),
        "state_initialiser": si.DeterministicCRWStateInitialiser(state),
        "job_conservation_flag": True,
        "list_boundary_constraint_matrices": None,
    }


def test_random_heuristic_agent_starving():
    # Single server queue
    safety_stock = 10.0
    state = 5 * np.ones((1, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 1))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 1))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert action == np.zeros((1, 1))


def test_random_heuristic_agent():
    # Single server queue
    safety_stock = 1.0
    state = 1.1 * np.ones((1, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 1))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 1))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert action == np.ones((1, 1))


def test_random_heuristic_agent_multiple_buffers_eye_condition_starving():
    # Station scheduling three buffers, each of them having to be above safety stock
    safety_stock = 10.0
    state = 5 * np.ones((3, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 3))
    env_params["list_boundary_constraint_matrices"] = [np.eye(3)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.zeros((3, 1)))


def test_random_heuristic_agent_multiple_buffers_eye_condition():
    # Station scheduling three buffers, each of them having to be above safety stock
    safety_stock = 1.0
    state = 1.1 * np.ones((3, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 3))
    env_params["list_boundary_constraint_matrices"] = [np.eye(3)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action) == 1


def test_random_heuristic_agent_multiple_buffers_sum_condition_starving():
    # Station scheduling three buffers, the sum of their size having to be above safety stock
    safety_stock = 10.0
    state = 3 * np.ones((3, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 3))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 3))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.zeros((3, 1)))


def test_random_heuristic_agent_multiple_buffers_sum_condition():
    # Station scheduling three buffers, the sum of their size having to be above safety stock
    safety_stock = 10.0
    state = 5 * np.ones((3, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.ones((1, 3))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 3))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action) == 1


def test_random_heuristic_agent_multiple_buffers_multiple_resources_eye_cond_starving():
    # Two stations, each one scheduling two buffers, each of them having to be above safety stock.
    safety_stock = 10.0
    state = 5 * np.ones((4, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.hstack((np.eye(2), np.zeros((2, 2)))),
                                                       np.hstack((np.zeros((2, 2)), np.eye(2)))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.zeros((4, 1)))


def test_random_heuristic_agent_multiple_buffers_multiple_resources_eye_cond():
    # Two stations, each one scheduling two buffers, each of them having to be above safety stock.
    safety_stock = 9.9
    state = 10 * np.ones((4, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.hstack((np.eye(2), np.zeros((2, 2)))),
                                                       np.hstack((np.zeros((2, 2)), np.eye(2)))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action) == 2


def test_random_heuristic_agent_multiple_buffers_multiple_resources_sum_cond_starving():
    # Two stations, each one scheduling two buffers, the sum of their size having to be above
    # safety stock.
    safety_stock = 10
    state = 4 * np.ones((4, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.zeros((4, 1)))


def test_random_heuristic_agent_multiple_buffers_multiple_resources_sum_cond():
    # Two stations, each one scheduling two buffers, the sum of their size having to be above safety
    # stock.
    safety_stock = 9.9
    state = 5 * np.ones((4, 1))
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action) == 2


def test_random_heuristic_agent_multiple_buffers_multiple_resources_sum_cond_1_starve():
    # Two stations, each one scheduling two buffers, the sum of their size having to be above safety
    # stock.
    safety_stock = 9.9
    state = np.array([4, 5, 5, 5])[:, None]
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action[2:4]) == 1 and np.all(action[0:2] == np.zeros((2, 1)))


def test_random_heuristic_agent_multiple_buffers_multiple_resources_sum_cond_2_starve():
    # Two stations, each one scheduling two buffers, the sum of their size having to be above safety
    # stock.
    safety_stock = 9.9
    state = np.array([5, 5, 5, 4])[:, None]
    env_params = get_null_env_params(state)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = random_nonidling_agent.RandomNonIdlingAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action[0:2]) == 1 and np.all(action[2:4] == np.zeros((2, 1)))


def test_priority_nonidling_heuristic_agent_starving():
    # Single server queue
    buffer_processing_matrix = - np.ones((1, 1))
    safety_stock = 10.0
    state = 5 * np.ones((1, 1))
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 1))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 1))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert action == np.zeros((1, 1))


def test_priority_nonidling_heuristic_agent():
    # Single server queue
    buffer_processing_matrix = - np.ones((1, 1))
    safety_stock = 4.0
    state = 5 * np.ones((1, 1))
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 1))
    env_params["list_boundary_constraint_matrices"] = [np.ones((1, 1))]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock, name="LPAAgent")
    action = agent.map_state_to_actions(state)
    assert action == np.ones((1, 1))


def test_priority_nonidling_heuristic_agent_multiple_buffers_eye_condition_starving():
    # One station scheduling two buffers, one larger than the other, but both below safety stock.
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([9, 5])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.zeros((2, 1)))


def test_priority_nonidling_heuristic_agent_multiple_buffers_eye_condition_small_one_starve():
    # One station scheduling two buffers, one larger than the other. Only the large one is above
    # safety stock.
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([9, 11])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([0, 1])[:, None])


def test_priority_nonidling_heuristic_agent_multi_buffers_eye_cond_small_one_starve_reverse_ord():
    # One station scheduling two buffers, one larger than the other. Only the large one is above
    # safety stock, swap order with respect to previous test.
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([11, 10])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([1, 0])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_buffers_eye_condition():
    # One station scheduling two buffers, one larger than the other, both above safety stock.
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([30, 20])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([1, 0])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_buffers_eye_condition_reverse_order():
    # One station scheduling two buffers, one larger than the other, both above safety stock (swap
    # order with previous test).
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([20, 30])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([0, 1])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_largest_buffers_eye_condition():
    # One station scheduling two buffers, both equal and above safety stock.
    buffer_processing_matrix = - np.eye(2)
    safety_stock = 10.0
    state = np.array([11, 11])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.ones((1, 2))
    env_params["list_boundary_constraint_matrices"] = [np.eye(2)]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action) == 1


def test_priority_nonidling_heuristic_agent_multiple_buffers_multiple_resources_sum_cond():
    # Two stations, each one scheduling two buffers. The stations are connected in serial, such that
    # buffer 1 is connected with buffer 3, and 2 with 4.
    # Kind of condition doesn't matter since the largest buffer has to be above safety stock in this
    # agent.
    buffer_processing_matrix = np.array([[-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, -1, 0],
                                         [0, 1, 0, -1]])
    safety_stock = 10
    state = np.array([30, 20, 20, 30])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([1, 0, 0, 1])[:, None])


def test_priority_nonidling_heuristic_agent_multi_buffers_and_resources_sum_cond_reverse_order():
    # Two stations, each one scheduling two buffers. The stations are connected in serial, such that
    # buffer 1 is connected with buffer 3, and 2 with 4.
    # Kind of condition doesn't matter since the largest buffer has to be above safety stock in this
    # agent.
    buffer_processing_matrix = np.array([[-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, -1, 0],
                                         [0, 1, 0, -1]])
    safety_stock = 10
    state = np.array([20, 30, 30, 20])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([0, 1, 1, 0])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_buffers_and_resources_sum_cond_2_starve():
    # Two stations, each one scheduling two buffers. The stations are connected in serial, such that
    # buffer 1 is connected with buffer 3, and 2 with 4.
    # Kind of condition doesn't matter since the largest buffer has to be above safety stock in this
    # agent.
    buffer_processing_matrix = np.array([[-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, -1, 0],
                                         [0, 1, 0, -1]])
    safety_stock = 10
    state = np.array([30, 20, 9, 5])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.all(action == np.array([1, 0, 0, 0])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_largest_buffers_multiple_resources_sum_cond():
    # Two stations, each one scheduling two buffers. The stations are connected in serial, such that
    # buffer 1 is connected with buffer 3, and 2 with 4.
    # Kind of condition doesn't matter since the largest buffer has to be above safety stock in this
    # agent.
    buffer_processing_matrix = np.array([[-1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, -1, 0],
                                         [0, 1, 0, -1]])
    safety_stock = 10
    state = np.array([30, 30, 9, 5])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert np.sum(action[0:2]) == 1 and np.all(action[2:4] == np.array([0, 0])[:, None])


def test_priority_nonidling_heuristic_agent_multiple_activities_buffers_and_resources():
    # Two stations, each one scheduling two buffers. The stations are connected in serial, such that
    # buffer 1 is connected with buffer 3, and 2 with 4.
    # Kind of condition doesn't matter since the largest buffer has to be above safety stock in this
    # agent.
    buffer_processing_matrix = np.array([[-1, 0, -1, 0],
                                         [0, -1, 0, 0],
                                         [1, 0, -1, 0],
                                         [0, 1, 0, -1]])
    safety_stock = 10
    state = np.array([30, 20, 5, 20])[:, None]
    env_params = get_null_env_params(
        state, buffer_processing_matrix=buffer_processing_matrix)
    env_params["constituency_matrix"] = np.array([[1, 1, 1, 0], [0, 0, 1, 1]])
    env_params["list_boundary_constraint_matrices"] = [np.array([[1, 1, 0, 0], [0, 0, 1, 0]]),
                                                       np.array([[0, 0, 1, 1]])]

    env = crw.ControlledRandomWalk(**env_params)
    agent = longest_priority_agent.LongestBufferPriorityAgent(env, safety_stock)
    action = agent.map_state_to_actions(state)
    assert (action[0] + action[2] == 1) and (action[1]
                                             == 0) and (action[3] == 1)


def test_priority_heuristic_agent_init_all_resources_given():
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    assert agent.priorities == priorities


def test_priority_heuristic_agent_init_not_all_resources_given():
    priorities = {0: 0, 2: 5}
    expected_priorities = {0: 0, 1: None, 2: 5}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    assert agent.priorities == expected_priorities


def test_priority_heuristic_agent_init_wrong_activity_given():
    priorities = {0: 0, 2: 1}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    with pytest.raises(AssertionError):
        _ = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)


def test_priority_heuristic_agent_sample_random_action_empty_possible_actions():
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [10.], [0.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    action = np.array([[1], [0], [1], [0], [0], [0], [0]])
    activities = np.array([3, 4, 5, 6])
    updated_action = agent.sample_random_actions(state=state, action=action, activities=activities)
    assert np.all(action == updated_action)


def test_priority_heuristic_agent_sample_random_action_one_possible_action():
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [0.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., -1., 0., 0., 0., 0.],
                                         [0., -1., 0., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    action = np.array([[1], [0], [0], [0], [1], [0], [0]])
    expected_action = np.array([[1], [0], [1], [0], [1], [0], [0]])
    activities = np.array([1, 2])
    updated_action = agent.sample_random_actions(state=state, action=action, activities=activities)
    assert np.all(expected_action == updated_action)


def test_priority_heuristic_agent_sample_random_action_multiple_possible_actions():
    np.random.seed(42)
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    action = np.array([[1], [0], [1], [0], [0], [0], [0]])
    expected_action = np.array([[1], [0], [1], [0.25], [0.25], [0.25], [0.25]])
    activities = np.array([3, 4, 5, 6])
    num_sim = int(1e4)
    updated_action = np.zeros((buffer_processing_matrix.shape[1], num_sim))
    for i in np.arange(num_sim):
        updated_action[:, [i]] = agent.sample_random_actions(state=state, action=action,
                                                             activities=activities)
    average_updated_action = np.sum(updated_action, axis=1) / float(num_sim)
    np.testing.assert_array_almost_equal(average_updated_action.reshape(-1, 1), expected_action,
                                         decimal=2)


def test_priority_heuristic_agent_map_state_to_actions_no_priorities():
    np.random.seed(42)
    priorities = {}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    expected_action = np.array([[1], [0.5], [0.5], [0.25], [0.25], [0.25], [0.25]])
    num_sim = int(1e4)
    actions = np.zeros((buffer_processing_matrix.shape[1], num_sim))
    for i in np.arange(num_sim):
        actions[:, [i]] = agent.map_state_to_actions(state=state)
    average_action = np.sum(actions, axis=1) / float(num_sim)
    np.testing.assert_array_almost_equal(average_action.reshape(-1, 1), expected_action,
                                         decimal=2)


def test_priority_heuristic_agent_map_state_to_actions_full_priorities_empty_buffer():
    np.random.seed(41)
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [10.], [0.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., -1., -1., 0., -1.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., 0., 0., -1., 0.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    constituency_matrix_original = constituency_matrix.copy()
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    expected_average_action = np.array([[1], [0.], [1.], [0.33], [0.33], [0.], [0.33]])
    num_sim = 5e4
    actions = np.zeros((buffer_processing_matrix.shape[1], int(num_sim)))
    for i in np.arange(int(num_sim)):
        actions[:, [i]] = agent.map_state_to_actions(state=state)
    average_action = np.sum(actions, axis=1) / num_sim
    np.testing.assert_array_almost_equal(average_action.reshape(-1, 1), expected_average_action,
                                         decimal=2)
    assert np.all(constituency_matrix_original == constituency_matrix)
    assert np.all(constituency_matrix_original == env.constituency_matrix)


def test_priority_heuristic_agent_map_state_to_actions_full_priorities_full_buffer():
    priorities = {0: 0, 1: 2, 2: 5}
    state = np.array([[10.], [10.], [10.]])
    buffer_processing_matrix = np.array([[-1., 0., 0., 0., 0., 0., 0.],
                                         [0., -1., -1., 0., 0., 0., 0.],
                                         [0., 0., 0., -1., -1., -1., -1.]])
    constituency_matrix = np.array([[1., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 0., 1., 1., 1., 1.]])
    env_params = get_null_env_params(state=state, buffer_processing_matrix=buffer_processing_matrix,
                                     constituency_matrix=constituency_matrix)

    env = crw.ControlledRandomWalk(**env_params)
    agent = custom_priority_agent.CustomActivityPriorityAgent(env, priorities)

    expected_action = np.array([[1], [0], [1], [0], [0], [1], [0]])
    action = agent.map_state_to_actions(state=state)
    assert np.all(action == expected_action)
