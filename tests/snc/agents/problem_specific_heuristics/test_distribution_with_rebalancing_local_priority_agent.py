import numpy as np
import pytest
import src.snc.environments.examples_distribution_with_rebalancing as \
    examples_distribution_with_rebalancing
from src import snc as dwr_agent, snc as ps

# dwr: distribution with rebalancing


REFILL_SUPPLY_ACTIONS = ['act1', 'act7', 'act13']
DRAIN_DEMAND_ACTIONS = ['act2', 'act8', 'act14']
REFILL_WAREHOUSE_ACTIONS = ['act3', 'act9', 'act15']
NO_ACTIONS = []  # List[str]

TEST_SAFETY_STOCK = np.array([100, 10, 0, 100, 10, 0, 100, 10, 0])[:, None]


def get_distribution_with_rebalancing_example_env(initial_state, seed=42):
    return examples_distribution_with_rebalancing.three_warehouses_simplified(
        d1=2.5, d2=2.5, d3=2.5,
        mu1=10, mu2=100, mu3=2.5, mu5=0.1, mu6=None,
        mu7=10, mu8=100, mu9=3.8, mu11=5, mu12=None,
        mu13=10, mu14=100, mu15=2.5, mu17=0.1, mu18=None,
        mu19=0.1, mu20=5, mu21=0.1,
        cost_per_buffer=np.array(
            [[1], [1], [5], [1], [1], [5], [1], [1], [5]]),
        initial_state=initial_state,
        job_conservation_flag=True,
        job_gen_seed=seed,
        r_to_w_rebalance=False
    )


def run_greedy_actions_experiment(state, safety_stock, expected_action):
    env = get_distribution_with_rebalancing_example_env(state)
    agent = dwr_agent.DistributionWithRebalancingLocalPriorityAgent(env, safety_stock)

    action = agent.get_greedy_actions_no_rebalancing(state)
    assert np.all(set(expected_action) == action)


def run_rebalancing_actions_experiment(state, safety_stock, greedy_actions, expected_action):
    env = get_distribution_with_rebalancing_example_env(state)
    agent = dwr_agent.DistributionWithRebalancingLocalPriorityAgent(env, safety_stock)

    action = agent.get_rebalancing_actions(state, greedy_actions)
    assert np.all(set(expected_action) == action)


def test_greedy_actions_no_rebalancing__all_surplus():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = safety_stock + 5

    # if all surplus - drain the demand buffers
    expected_action = DRAIN_DEMAND_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__all_deficit():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = np.zeros_like(safety_stock)

    # if deficit, take all feedforward actions
    expected_action = REFILL_WAREHOUSE_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__at_safety_stock_levels():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()

    # expect no actions if at safety stock levels
    expected_action = NO_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_a_supply_buffer():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[1, 0] = 0
    state[2, 0] = 100

    # if deficit at a supply buffers, feedforward in that channel
    expected_action = [REFILL_SUPPLY_ACTIONS[0], REFILL_WAREHOUSE_ACTIONS[0]]
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_a_supply_buffer_surplus_at_warehouse():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[1, 0] = 0
    state[0, 0] *= 2

    # if deficit at a supply buffers, feedforward in that channel
    expected_action = [REFILL_SUPPLY_ACTIONS[0]]
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_all_supply_buffers():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[[1, 4, 7], 0] = 0

    # if deficit at all supply buffers, feedforward all channels
    expected_action = REFILL_SUPPLY_ACTIONS + REFILL_WAREHOUSE_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_all_supply_buffers_surplus_at_warehouses():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[[1, 4, 7], 0] = 0
    state[[0, 3, 6], 0] *= 2

    # if deficit at all supply buffers, feedforward all channels
    expected_action = REFILL_SUPPLY_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_a_warehouse():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0, 0] = 0

    # if deficit at a warehouse, feedforward from retail buffer
    expected_action = [REFILL_WAREHOUSE_ACTIONS[0]]
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_greedy_actions_no_rebalancing__deficit_at_all_warehouses():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[[0, 3, 6], 0] = 0

    # if deficit at all warehouse, feedforward from retail buffer
    expected_action = REFILL_WAREHOUSE_ACTIONS
    run_greedy_actions_experiment(state, safety_stock, expected_action)


def test_adding_rebalancing_actions__no_rebalancing_necessary():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = safety_stock * 5

    # if all warehouse surplus - no rebalancing actions
    greedy_actions = []
    expected_action = NO_ACTIONS
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__no_rebalancing_possible():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = safety_stock / 5

    # if all deficit - no rebalancing actions
    greedy_actions = REFILL_SUPPLY_ACTIONS + REFILL_WAREHOUSE_ACTIONS
    expected_action = NO_ACTIONS
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__one_surplus_warehouse():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] -= 50
    state[3] += 50

    # if one deficit, one surplus - one refill action
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[0]]
    expected_action = ['act20'] + ['act15', 'act17']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__one_surplus_warehouse_reverse():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] += 50
    state[3] -= 50

    # if one deficit, one surplus - one refill action
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[0]]
    expected_action = ['act5'] + ['act15', 'act21']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__two_surplus_warehouses_1():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] -= 50
    state[3] += 50
    state[6] += 75

    # if one deficit, two surplus - two refill action
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[0]]
    expected_action = ['act17', 'act20']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__two_surplus_warehouses_2():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] += 50
    state[3] -= 50
    state[6] += 75

    # if one deficit, two surplus - one refill action
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[1]]
    expected_action = ['act5', 'act21']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__rebalance_to_greatest_deficit():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] += 50
    state[3] -= 50
    state[6] -= 75

    # if two deficit one surplus, refill to greatest deficit
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[1]]
    expected_action = ['act19']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__no_rebalance_due_to_conflict():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] += 50
    state[1] -= 5
    state[3] -= 50
    state[6] -= 75

    # if two deficit one surplus, refill to greatest deficit
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[0],
                      REFILL_WAREHOUSE_ACTIONS[1],
                      REFILL_SUPPLY_ACTIONS[0]]
    expected_action = []
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def test_adding_rebalancing_actions__no_rebalance_due_to_conflict_2():
    safety_stock = TEST_SAFETY_STOCK.copy()
    state = TEST_SAFETY_STOCK.copy()
    state[0] += 50
    state[1] -= 5
    state[3] -= 50

    # if two deficit one surplus, refill to greatest deficit
    greedy_actions = [REFILL_WAREHOUSE_ACTIONS[0],
                      REFILL_WAREHOUSE_ACTIONS[1],
                      REFILL_SUPPLY_ACTIONS[0]]
    expected_action = ['act21', 'act15']
    run_rebalancing_actions_experiment(
        state, safety_stock, greedy_actions, expected_action)


def get_map_state_to_action_test_params():
    return [(
        '1.all_surplus_all_drain',
        [200, 200, 200, 200, 200, 200, 200, 200, 200],
        [0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    ), (
        '2.no_rebalance_one_supply_refill_all_drain',
        [200, 2, 200, 200, 200, 200, 200, 200, 200],
        [1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
    ), (
        '3.no_rebalance_one_supply_refill_one_drain',
        [200, 2, 0, 200, 200, 200, 200, 200, 0],
        [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    ), (
        '4.one_rebalance_one_supply_refill_one_drain',
        [200, 2, 0, 50, 200, 200, 200, 200, 0],
        [1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.]
    ), (
        '5.two_rebalance_no_supply_refill_one_drain',
        [50, 200, 0, 200, 200, 200, 200, 200, 0],
        [0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.]
    )]


@pytest.mark.parametrize("test_name,state,expected_action", get_map_state_to_action_test_params())
def test_map_state_to_actions(test_name, state, expected_action):
    safety_stock = TEST_SAFETY_STOCK.copy()

    env = get_distribution_with_rebalancing_example_env(np.array(state)[:, None])
    agent = dwr_agent.DistributionWithRebalancingLocalPriorityAgent(env, safety_stock)

    action = agent.map_state_to_actions(np.array(state)[:, None])
    assert np.all(np.array(expected_action)[:, None] == action)


def test_distribution_with_rebalancing_integration():
    initial_state = np.array([0, 0, 10, 0, 0, 0, 0, 0, 10])[:, None]
    safety_stock = np.array([10, 5, 0, 10, 5, 0, 10, 5, 0])[:, None]

    env = get_distribution_with_rebalancing_example_env(np.array(initial_state)[:, None])
    agent = dwr_agent.DistributionWithRebalancingLocalPriorityAgent(env, safety_stock)
    simulator = ps.SncSimulator(env, agent, discount_factor=0.95)

    num_simulation_steps = 1000
    simulator.run(num_simulation_steps=num_simulation_steps)
