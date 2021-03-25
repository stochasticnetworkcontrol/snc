from collections import deque, defaultdict
import numpy as np
import pytest

from snc.environments.closed_loop_crw import ClosedLoopCRW
import snc.environments.examples as examples
from snc.environments.job_generators.discrete_review_job_generator import \
    DeterministicDiscreteReviewJobGenerator
from snc.environments.job_generators.scaled_bernoulli_services_poisson_arrivals_generator import \
    ScaledBernoulliServicesPoissonArrivalsGenerator
from snc.environments.state_initialiser import DeterministicCRWStateInitialiser


def build_closed_loop_env_2_demand_buffers(
        demand_to_supplier_routes,
        constituency_matrix,
        initial_state=np.zeros((5, 1))
):
    ind_surplus_buffers = [1, 3]
    job_gen_seed = 42
    mu = 1.5
    mud = 3
    mus = 1.5
    alpha = 0.95
    cost_per_buffer = np.array([[1], [2], [5], [3], [8]])
    demand_rate = np.array([[0], [0], [alpha], [0], [alpha]])
    buffer_processing_matrix = np.array([[-mu, -mu/3,    0, mus,    0, 0],
                                         [2*mu/3,  0, -mud,   0,    0, 0],
                                         [0,       0, -mud,   0,    0, 0],
                                         [mu/3, mu/3,    0,   0, -mud, mus/3],
                                         [0,       0,    0,   0, -mud, 0]])
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    cl_env = ClosedLoopCRW(
        demand_to_supplier_routes,
        ind_surplus_buffers,
        cost_per_buffer,
        np.ones_like(demand_rate) * np.inf,
        constituency_matrix,
        job_generator,
        state_initialiser
    )
    return cl_env


def build_closed_loop_single_station_demand_model(initial_state=np.zeros((3, 1)), toa=100):
    ind_surplus_buffers = [1]
    demand_to_supplier_routes = {2: (2, toa)}
    job_gen_seed = 42
    mu = 3
    mud = 3
    mus = 3
    alpha = 2
    cost_per_buffer = np.array([[1], [2], [5]])
    demand_rate = np.array([[0], [0], [alpha]])
    buffer_processing_matrix = np.array([[-mu, 0, mus],
                                         [mu, -mud, 0],
                                         [0, -mud, 0]])
    job_generator = DeterministicDiscreteReviewJobGenerator(demand_rate,
                                                            buffer_processing_matrix,
                                                            job_gen_seed,
                                                            sim_time_interval=1)
    constituency_matrix = np.eye(3)
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    cl_env = ClosedLoopCRW(
        demand_to_supplier_routes,
        ind_surplus_buffers,
        cost_per_buffer,
        np.ones_like(demand_rate) * np.inf,
        constituency_matrix,
        job_generator,
        state_initialiser
    )
    return cl_env


def test_get_supply_and_demand_ids():
    demand = (0, 1, 2, 3, 4, 5, 6, 7)
    supply = (10, 11, 12, 13, 14, 15, 16, 17)
    toa = (20, 21, 22, 23, 24, 25, 26, 27)
    demand_to_supplier_routes = {demand[i]: (supply[i], toa[i]) for i in range(8)}

    supply_id, demand_id = ClosedLoopCRW.get_supply_and_demand_ids(demand_to_supplier_routes)
    assert supply_id == list(supply)
    assert demand_id == list(demand)


def test_are_demand_ids_unique():
    demand_id = list(range(4))
    assert ClosedLoopCRW.are_demand_ids_unique(demand_id)


def test_are_demand_ids_unique_false():
    demand_id = [0, 0, 1, 2]
    assert not ClosedLoopCRW.are_demand_ids_unique(demand_id)


def test_get_supply_and_demand_ids_repeated_ids():
    demand = (0, 1, 2, 3, 4)
    supply = (10, 11, 10, 11, 12)
    toa = (20, 21, 20, 21, 22)
    demand_to_supplier_routes = {demand[i]: (supply[i], toa[i]) for i in range(len(demand))}

    supply_id, demand_id = ClosedLoopCRW.get_supply_and_demand_ids(demand_to_supplier_routes)
    assert supply_id == [10, 11, 12]
    assert demand_id == list(demand)


@pytest.mark.parametrize('supply_ids,demand_ids,env_class', [
    ([3], [5], examples.double_reentrant_line_with_demand_only_shared_resources_model),
    ([7, 8], [14, 15], examples.complex_demand_driven_model),
])
def test_is_demand_to_supplier_routes_consistent_with_job_generator_envs(
        supply_ids,
        demand_ids,
        env_class
):
    env = env_class()
    assert ClosedLoopCRW.is_demand_to_supplier_routes_consistent_with_job_generator(
        supply_ids,
        demand_ids,
        env.constituency_matrix,
        env.job_generator.supply_nodes,
        env.job_generator.demand_nodes.values()
    )


def test_is_supply_ids_consistent_with_job_generator():
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    assert ClosedLoopCRW.is_supply_ids_consistent_with_job_generator(
        env.supply_ids,
        env.job_generator.supply_nodes,
        env.constituency_matrix
    )


def test_is_supply_ids_consistent_with_job_generator_false():
    supply_ids = [2, 3]  # It should be [2, 4]
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    assert not ClosedLoopCRW.is_supply_ids_consistent_with_job_generator(
        supply_ids,
        env.job_generator.supply_nodes,
        env.constituency_matrix
    )


def test_initialise_supply_buffers():
    supply_id = [0, 11]
    supply_buf = ClosedLoopCRW.initialise_supply_buffers(supply_id)
    assert supply_buf == {0: 0, 11: 0}


def test_get_activity_supply_resource_association_eye():
    supply_nodes = [(0, 1), (1, 2)]
    constituency_matrix = np.eye(4)
    activity_to_resource, resource_to_activity = \
        ClosedLoopCRW.get_activity_supply_resource_association(
            supply_nodes,
            constituency_matrix
        )
    assert activity_to_resource == {1: 1, 2: 2}
    assert resource_to_activity == {1: [1], 2: [2]}


def test_get_activity_supply_resource_association_only_one_resource():
    supply_nodes = [(0, 1), (1, 2)]
    constituency_matrix = np.array([[0, 1, 1],
                                    [1, 0, 0]])
    activity_to_resource, resource_to_activity = \
        ClosedLoopCRW.get_activity_supply_resource_association(
            supply_nodes,
            constituency_matrix
        )
    assert activity_to_resource == {1: 0, 2: 0}
    assert resource_to_activity == {0: [1, 2]}


def test_get_activity_supply_resource_association_two_resources():
    supply_nodes = [(0, 1), (1, 2), (2, 0)]
    constituency_matrix = np.array([[1, 1, 0],
                                    [0, 0, 1]])
    activity_to_resource, resource_to_activity = \
        ClosedLoopCRW.get_activity_supply_resource_association(
            supply_nodes,
            constituency_matrix
        )
    assert activity_to_resource == {0: 0, 1: 0, 2: 1}
    assert resource_to_activity == {0: [0, 1], 1: [2]}


def test_get_activity_supply_resource_association_action_belongs_to_two_resources():
    supply_nodes = [(0, 1)]
    constituency_matrix = np.array([[1, 1],
                                    [0, 1]])
    with pytest.raises(AssertionError):
        _, _ = ClosedLoopCRW.get_activity_supply_resource_association(
            supply_nodes,
            constituency_matrix
        )


def test_get_supply_activity_to_buffer_association_only_one_supply_activity():
    supply_nodes = [(0, 2)]
    activity_to_buffer = ClosedLoopCRW.get_supply_activity_to_buffer_association(supply_nodes)
    assert activity_to_buffer == {2: 0}


def test_get_supply_activity_to_buffer_association_multiple_supply_activities():
    supply_nodes = [(0, 2), (1, 3)]
    activity_to_buffer = ClosedLoopCRW.get_supply_activity_to_buffer_association(supply_nodes)
    assert activity_to_buffer == {2: 0, 3: 1}


@pytest.mark.parametrize(
    's1,s2', [(0, 0), (3, 1), (10, 20)]
)
def test_sum_supplier_outbound(s1, s2):
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    routing_matrix = np.zeros_like(env.job_generator.buffer_processing_matrix)
    routing_matrix[0, 3] = s1
    routing_matrix[3, 5] = s2
    sum_outbound = env.sum_supplier_outbound(routing_matrix)
    assert sum_outbound == {2: s1, 4: s2}


@pytest.mark.parametrize(
    's1,s2', [(0, 0), (3, 1), (10, 20)]
)
def test_sum_supplier_outbound_one_resource_multiple_routes(s1, s2):
    demand_to_supplier_routes = {2: (2, 100), 4: (2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    routing_matrix = np.zeros_like(env.job_generator.buffer_processing_matrix)
    routing_matrix[0, 3] = s1
    routing_matrix[3, 5] = s2
    sum_outbound = env.sum_supplier_outbound(routing_matrix)
    assert sum_outbound == {2: s1 + s2}


def get_truncated_val(s, a):
    return s if s < a else a


@pytest.mark.parametrize(
    's1,s2,a1,a2',
    [
        (0, 0, 0, 0),  # Empty and none available.
        (0, 0, 1, 1),  # Empty but available.
        (3, 2, 0, 0),  # Some but none available.
        (3, 2, 3, 2),  # Exactly what's available.
        (3, 2, 2, 1),  # More than available.
        (3, 2, 4, 3),  # Less than available.
    ]
)
def test_truncate_routing_matrix_supplier(s1, s2, a1, a2):
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    routing_matrix = np.zeros_like(env.job_generator.buffer_processing_matrix)
    routing_matrix[0, 3] = s1
    routing_matrix[3, 5] = s2
    env.supply_buffers[2] = a1
    env.supply_buffers[4] = a2
    new_routing_matrix = env.truncate_routing_matrix_supplier(2, routing_matrix, a1)
    assert new_routing_matrix[0, 3] == get_truncated_val(s1, a1)
    new_routing_matrix = env.truncate_routing_matrix_supplier(4, new_routing_matrix, a2)
    assert new_routing_matrix[3, 5] == get_truncated_val(s2, a2)


@pytest.mark.parametrize(
    's1,s2,a',
    [
        (0, 0, 0),  # Empty and none available.
        (0, 0, 1),  # Empty but available.
        (3, 2, 0),  # Some but none available.
        (3, 2, 5),  # Exactly what's available.
        (3, 2, 4),  # More than available.
        (3, 2, 6),  # Less than available.
    ]
)
def test_truncate_routing_matrix_supplier_one_resource_multiple_routes(s1, s2, a):
    demand_to_supplier_routes = {2: (2, 100), 4: (2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    routing_matrix = np.zeros_like(env.job_generator.buffer_processing_matrix)
    routing_matrix[0, 3] = s1
    routing_matrix[3, 5] = s2
    env.supply_buffers[2] = a
    new_routing_matrix = env.truncate_routing_matrix_supplier(2, routing_matrix, a)
    assert new_routing_matrix[0, 3] + new_routing_matrix[3, 5] == get_truncated_val(s1 + s2, a)


def get_new_supply_buffers(s, a):
    return a - s if s < a else 0


@pytest.mark.parametrize(
    's1,s2,a1,a2',
    [
        (0, 0, 0, 0),  # Empty and none available.
        (0, 0, 1, 1),  # Empty but available.
        (3, 2, 0, 0),  # Some but none available.
        (3, 2, 3, 2),  # Exactly what's available.
        (3, 2, 2, 1),  # More than available.
        (3, 2, 4, 3),  # Less than available.
    ]
)
def test_ensure_jobs_conservation(s1, s2, a1, a2):
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    state_plus_arrivals = np.zeros((5, 1))
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    routing_matrix = np.zeros_like(env.job_generator.buffer_processing_matrix)
    routing_matrix[0, 3] = s1
    routing_matrix[3, 5] = s2
    env.supply_buffers[2] = a1
    env.supply_buffers[4] = a2
    new_routing_matrix = env.ensure_jobs_conservation(routing_matrix, state_plus_arrivals)
    assert new_routing_matrix[0, 3] == get_truncated_val(s1, a1)
    assert new_routing_matrix[3, 5] == get_truncated_val(s2, a2)
    assert env.supply_buffers[2] == get_new_supply_buffers(s1, a1)
    assert env.supply_buffers[4] == get_new_supply_buffers(s2, a2)


def test_ensure_jobs_conservation_with_enough_jobs():
    state = 3 * np.ones((3, 1))
    routing_matrix = np.array([[-3, 0, 3],
                               [3, -3, 0],
                               [0, -3, 0]])

    env = build_closed_loop_single_station_demand_model()
    new_routing_jobs_matrix = env.ensure_jobs_conservation(routing_matrix, state)
    assert np.all(new_routing_jobs_matrix == routing_matrix)


def test_ensure_jobs_conservation_with_not_enough_jobs():
    state = np.array([[2], [1], [2]])
    routing_matrix = np.array([[-3, 0, 3],
                               [3, -3, 0],
                               [0, -3, 0]])

    env = build_closed_loop_single_station_demand_model()
    env.supply_buffers[2] = 1

    expected_routing_matrix = np.array([[-2, 0, 1],
                                        [2, -1, 0],
                                        [0, -1, 0]])

    new_routing_jobs_matrix = env.ensure_jobs_conservation(routing_matrix, state)
    assert np.all(new_routing_jobs_matrix == expected_routing_matrix)


def test_ensure_jobs_conservation_with_zero_jobs():
    state = np.zeros((3, 1))
    routing_matrix = np.array([[-3, 0, 3],
                               [3, -3, 0],
                               [0, -3, 0]])

    env = build_closed_loop_single_station_demand_model()
    env.supply_buffers[2] = 1

    expected_routing_matrix = np.array([[0, 0, 1],
                                        [0, 0, 0],
                                        [0, 0, 0]])
    new_routing_jobs_matrix = env.ensure_jobs_conservation(routing_matrix, state)
    assert np.all(new_routing_jobs_matrix == expected_routing_matrix)


def test_get_num_items_supply_buff():
    demand_to_supplier_routes = {2: (2, 100), 4: (2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    env.supply_buffers[2] = 10
    env.supply_buffers[4] = 20
    assert env.get_num_items_supply_buff() == 30


def test_get_num_items_supply_buff_init():
    demand_to_supplier_routes = {2: (2, 100), 4: (2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1],
                                    [0, 0, 0, 0, 1, 0]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    assert env.get_num_items_supply_buff() == 0


def test_get_num_items_in_transit_to_suppliers():
    supp1 = 2
    supp2 = 4
    demand_to_supplier_routes = {2: (supp1, 100), 4: (supp2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    toa1 = 10
    toa2 = 11
    env.in_transit_parcels[toa1].append((supp1, 1))
    env.in_transit_parcels[toa2].append((supp2, 7))
    assert env.get_num_items_in_transit_to_suppliers() == 8


def test_get_num_items_in_transit_to_suppliers_multiple_in_transit():
    supp1 = 2
    supp2 = 4
    demand_to_supplier_routes = {2: (supp1, 100), 4: (supp2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    toa1 = 10
    toa2 = 11
    env.in_transit_parcels[toa1].extend([(supp1, 9), (supp1, 1)])
    env.in_transit_parcels[toa2].extend([(supp2, 20), (supp1, 10)])
    assert env.get_num_items_in_transit_to_suppliers() == 40


def test_get_num_items_in_transit_to_suppliers_init():
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    assert env.get_num_items_in_transit_to_suppliers() == 0


def test_assert_remains_closed_network_empty():
    initial_state = np.zeros((5, 1))
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(
        demand_to_supplier_routes,
        constituency_matrix,
        initial_state
    )
    env.assert_remains_closed_network()


def test_assert_remains_closed_network_false():
    initial_state = np.ones((5, 1))
    demand_to_supplier_routes = {2: (2, 100), 4: (4, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(
        demand_to_supplier_routes,
        constituency_matrix,
        initial_state
    )
    env.state[0] = 0  # Remove one item without putting it anywhere else.
    with pytest.raises(AssertionError):
        env.assert_remains_closed_network()


def test_get_num_items_state_without_demand():
    initial_state = 5 * np.ones((5, 1))  # 25 items, 10 in demand buffers.
    supp1 = 2
    supp2 = 4
    demand_to_supplier_routes = {2: (supp1, 100), 4: (supp2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(
        demand_to_supplier_routes,
        constituency_matrix,
        initial_state
    )
    assert np.all(env.num_initial_items == 15)


def test_assert_remains_closed_network_all_in_transit_and_suppliers():
    initial_state = 5 * np.ones((5, 1))  # 25 items, 10 in demand buffers.
    supp1 = 2
    supp2 = 4
    demand_to_supplier_routes = {2: (supp1, 100), 4: (supp2, 300)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(
        demand_to_supplier_routes,
        constituency_matrix,
        initial_state
    )
    env.state = np.zeros((5, 1))
    env.supply_buffers[2] = 1
    env.supply_buffers[4] = 2
    toa1 = 10
    toa2 = 11
    env.in_transit_parcels[toa1].extend([(supp1, 3), (supp1, 1)])
    env.in_transit_parcels[toa2].extend([(supp2, 2), (supp1, 6)])
    env.assert_remains_closed_network()


def test_get_satisfied_demand():
    drained_amount = np.array([1, 2, 3, 4, 5])[:, None]
    demand_id = [0, 3]

    satisfied_demand = ClosedLoopCRW.get_satisfied_demand(drained_amount, demand_id)
    assert satisfied_demand == {0: 1, 3: 4}


def test_fill_in_transit_to_suppliers():
    initial_state = np.array([10, 4, 3])[:, None]
    toa = 200
    amount = 7
    current_time = 42
    env = build_closed_loop_single_station_demand_model(initial_state, toa)

    env._t = current_time
    satisfied_demand = {2: amount}  # From buffer 2, which will be delivered at resource 2.
    env.fill_in_transit_to_suppliers(satisfied_demand)
    assert env.in_transit_parcels == {current_time + toa: [(2, amount)]}


def test_fill_in_transit_to_suppliers_multiple_parcels():
    initial_state = np.array([10, 4, 3])[:, None]
    toa = 200
    amount1 = 7
    amount2 = 14
    current_time = 42
    env = build_closed_loop_single_station_demand_model(initial_state, toa)

    env._t = current_time
    env.fill_in_transit_to_suppliers({2: amount1})
    env.fill_in_transit_to_suppliers({2: amount2})
    assert env.in_transit_parcels == {current_time + toa: [(2, amount1), (2, amount2)]}


def test_fill_in_transit_to_suppliers_multiple_simultaneous_parcels():
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    current_time = 42
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)

    env._t = current_time
    env.fill_in_transit_to_suppliers({2: 10, 4: 13})

    assert env.in_transit_parcels == {
        current_time + toa2: [(2, 10)],
        current_time + toa4: [(4, 13)]
    }


def test_fill_in_transit_to_suppliers_multiple_resources_multiple_sequential_parcels():
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    current_time = 42
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)

    env._t = current_time
    env.fill_in_transit_to_suppliers({2: 10})
    env.fill_in_transit_to_suppliers({4: 13})

    env._t = current_time + 100
    env.fill_in_transit_to_suppliers({2: 14})

    assert env.in_transit_parcels == {
        current_time + toa2: [(2, 10)],
        current_time + toa4: [(4, 13)],
        current_time + toa2 + 100: [(2, 14)]
    }


def test_fill_supply_buffers_empty_in_transit():
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)

    env.fill_supply_buffers()
    assert env.supply_buffers == {2: 0, 4: 0}
    assert env.in_transit_parcels == defaultdict(list)


def test_fill_supply_buffers_some_in_transit_but_not_arrived():
    amount2 = 10
    amount4 = 11
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    env.fill_in_transit_to_suppliers({2: amount2, 4: amount4})

    env._t = toa2 - 1
    env.fill_supply_buffers()
    assert env.supply_buffers == {2: 0, 4: 0}
    assert env.in_transit_parcels == {toa2: [(2, amount2)], toa4: [(4, amount4)]}


def test_fill_supply_buffers_some_in_transit_only_one_arrived():
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    env.fill_in_transit_to_suppliers({2: 10})
    env.fill_in_transit_to_suppliers({4: 11})

    env._t = toa2
    env.fill_supply_buffers()
    assert env.supply_buffers == {2: 10, 4: 0}
    assert env.in_transit_parcels == {toa4: [(4, 11)]}


def test_fill_supply_buffers_some_in_transit_two_arrived():
    toa2 = 100
    toa4 = 300
    demand_to_supplier_routes = {2: (2, toa2), 4: (4, toa4)}
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    env = build_closed_loop_env_2_demand_buffers(demand_to_supplier_routes, constituency_matrix)
    env.fill_in_transit_to_suppliers({4: 11})

    env._t = 200
    env.fill_in_transit_to_suppliers({2: 10})

    env._t = 300
    env.fill_supply_buffers()
    assert env.supply_buffers == {2: 10, 4: 11}
    assert env.in_transit_parcels == defaultdict(list)


def test_step():
    env = build_closed_loop_single_station_demand_model(
        initial_state=np.array([[10], [5], [3]]),
        toa=100
    )
    action = np.array([[0], [1], [1]])
    env.step(action)
    # Nothing done in buffer 0. 3 are removed from buffers 1 and 2, but 2 new arrivals at buffer 2.
    assert np.all(env.state == np.array([[10], [2], [2]]))
    assert env.in_transit_parcels == {101: [(2, 3)]}  # Deliver 3 items to resource 2 at time 101.
    assert env.supply_buffers == {2: 0}


def test_step_many_steps():
    toa = 100
    env = build_closed_loop_single_station_demand_model(
        initial_state=np.array([[10], [5], [3]]),
        toa=toa
    )
    alpha = env.job_generator.demand_rate[2]
    action = np.array([[1], [1], [1]])

    env.step(action)
    assert np.all(env.state == np.array([[7], [5], [alpha]]))
    assert env.in_transit_parcels == {101: [(2, 3)]}
    assert env.supply_buffers == {2: 0}

    action = np.zeros((3, 1))
    for i in range(toa - 1):
        env.step(action)
        assert np.all(env.state == np.array([[7], [5], [alpha * env.t]]))
        assert env.in_transit_parcels == {101: [(2, 3)]}
        assert env.supply_buffers == {2: 0}

    env.step(action)
    assert np.all(env.state == np.array([[7], [5], [env.t * alpha]]))
    assert env.in_transit_parcels == defaultdict(list)
    assert env.supply_buffers == {2: 3}
