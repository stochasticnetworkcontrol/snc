# Set of examples of systems that can be modelled as controlled random walks, from CTCN book.

from typing import Optional
import numpy as np
from src.snc.environments \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
import src.snc.environments.state_initialiser as stinit
from src.snc.environments import ControlledRandomWalk


# This file contains examples for:
# --- Push AND pull examples
# - push_and_pull_model_minimal_example
# - push_and_pull_model_example
# - one_buffer_per_type_of_actor_with_simultaneous_actions_example
# - one_buffer_per_type_of_actor_with_single_action_example
# - two_demands_with_simultaneous_actions_example
# - one_side_with_simultaneous_actions_example
# - two_sides_with_simultaneous_actions_example
# --- Pull only examples with single resource per buffer
# - one_warehouse
# - two_warehouses
# - three_warehouses
# - three_warehouses_two_manufacturers_per_area
# --- Pull only simplified examples: no connection between supply and final product buffer
# - two_warehouses_simplified
# - three_warehouses_simplified
#
# Note: with the term ' distribution with rebalancing' we are referring to a network where:
# - a supplier produces items and can send them to one or more warehouses,
# - a warehouse can send items to one or more manufacturers,
# - a warehouse can send items to other warehouses
# - a manufacturer consumes the items

# Push AND pull examples---------------------------------------------------------------------------

def push_and_pull_model_minimal_example(d_in=2, d_out=2,
                                        mu=3,
                                        cost_per_buffer=np.ones((2, 1)),
                                        initial_state=(1, 1),
                                        capacity=np.ones((2, 1)) * np.inf,
                                        job_conservation_flag=True,
                                        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    """Station with one buffer and two routes, each one these resources representing a route to
    different buffers, each one in a different station."""
    # Problem data.
    demand_rate = np.array([d_in, d_out])[:, None]
    buffer_processing_matrix = np.array([[-mu],
                                         [-mu]])
    constituency_matrix = np.array([[1]])
    list_boundary_constraint_matrices = [np.array([[0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def push_and_pull_model_example(d_in=2, d_out=2,
                                mu=3, mud=10,
                                cost_per_buffer=np.ones((3, 1)),
                                initial_state=(1, 1, 1),
                                capacity=np.ones((3, 1)) * np.inf,
                                job_conservation_flag=True,
                                job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([d_in, 0, d_out])[:, None]
    buffer_processing_matrix = np.array([[-mu, 0],
                                         [mu, -mud],
                                         [0, -mud]])
    constituency_matrix = np.eye(2)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0]]),
                                         np.array([[0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def one_buffer_per_type_of_actor_with_simultaneous_activities_example(
        d_in=0.2, d_out=0.2,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1,
        cost_per_buffer=np.ones((4, 1)),
        initial_state=(1, 1, 1, 1),
        capacity=np.ones((4, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([d_in, 0, 0, d_out])[:, None]
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0, 0],
                                         [+mu1, 0, -mu3, 0],
                                         [0, +mu2, +mu3, -mu4],
                                         [0, 0, 0, -mu4]])
    constituency_matrix = np.eye(4)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1), (2, 2): (1, 2)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def one_buffer_per_type_of_actor_with_single_activity_example(
        d_in=0.2, d_out=0.2,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1,
        cost_per_buffer=np.ones((4, 1)),
        initial_state=(1, 1, 1, 1),
        capacity=np.ones((4, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([d_in, 0, 0, d_out])[:, None]
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0, 0],
                                         [+mu1, 0, -mu3, 0],
                                         [0, +mu2, +mu3, -mu4],
                                         [0, 0, 0, -mu4]])
    constituency_matrix = np.array([[1, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1), (2, 2): (1, 2)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def two_demands_with_simultaneous_activities_example(
        d_out_1=0.2, d_out_2=0.3,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1, mu5=0.1,
        cost_per_buffer=np.ones((5, 1)),
        initial_state=(1, 1, 1, 1, 1),
        capacity=np.ones((5, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, 0, d_out_1, d_out_2])[:, None]
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0, 0, +mu5],
                                         [+mu1, 0, -mu3, 0, 0],
                                         [0, +mu2, 0, -mu4, 0],
                                         [0, 0, -mu3, 0, 0],
                                         [0, 0, 0, -mu4, 0]])
    constituency_matrix = np.eye(5)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def one_side_with_simultaneous_activities_example(
        d_in_1=0.2, d_in_2=0.3, d_out_1=0.2, d_out_2=0.3,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1,
        mu5=0.1, mu6=0.1, mu7=0.1, mu8=0.1,
        cost_per_buffer=np.ones((7, 1)),
        initial_state=np.ones((7, 1)),
        capacity=np.ones((7, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([d_in_1, d_in_2, 0, 0, 0, d_out_1, d_out_2])[:, None]
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0, 0, 0, 0, 0, 0],
                                         [0, 0, -mu3, -mu4, 0, 0, 0, 0],
                                         [+mu1, 0, +mu3, 0, -mu5, -mu6, 0, 0],
                                         [0, +mu2, 0, 0, +mu5, 0, -mu7, 0],
                                         [0, 0, 0, +mu4, 0, +mu6, 0, -mu8],
                                         [0, 0, 0, 0, 0, 0, -mu7, 0],
                                         [0, 0, 0, 0, 0, 0, 0, -mu8]])
    constituency_matrix = np.eye(8)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(2, 0): (0, 0), (3, 1): (0, 1),
                                    (2, 2): (1, 2), (4, 3): (1, 3),
                                    (3, 4): (2, 4), (4, 5): (2, 5)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def two_sides_with_simultaneous_activities_example(
        d_in_1=0.2, d_in_2=0.3, d_in_3=0.2, d_in_4=0.3,
        d_out_1=0.05, d_out_2=0.05, d_out_3=0.05, d_out_4=0.05,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1, mu5=0.1, mu6=0.1, mu7=0.1, mu8=0.1,
        mu9=0.1, mu10=0.1, mu11=0.1, mu12=0.1, mu13=0.1, mu14=0.1, mu15=0.1, mu16=0.1,
        mu17=0.1, mu18=0.1, mu19=0.1, mu20=0.1, mu21=0.1, mu22=0.1,
        cost_per_buffer=np.ones((14, 1)),
        initial_state=np.ones((14, 1)),
        capacity=np.ones((14, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([d_in_1, d_in_2, 0, 0, 0, d_out_1, d_out_2,
                            d_in_3, d_in_4, 0, 0, 0, d_out_3, d_out_4])[:, None]
    buffer_processing_matrix = np.zeros((14, 22))
    buffer_processing_matrix[0:7, 0:8] = np.array([[-mu1, -mu2, 0, 0, 0, 0, 0, 0],
                                                   [0, 0, -mu3, -mu4, 0, 0, 0, 0],
                                                   [+mu1, 0, +mu3, 0, -mu5, -mu6, 0, 0],
                                                   [0, +mu2, 0, 0, +mu5, 0, -mu7, 0],
                                                   [0, 0, 0, +mu4, 0, +mu6, 0, -mu8],
                                                   [0, 0, 0, 0, 0, 0, -mu7, 0],
                                                   [0, 0, 0, 0, 0, 0, 0, -mu8]])
    buffer_processing_matrix[0, 8] = -mu9
    buffer_processing_matrix[1, 9] = -mu10
    buffer_processing_matrix[2, 10] = -mu11
    buffer_processing_matrix[9, 8:11] = (mu9, mu10, mu11)
    buffer_processing_matrix[7:14, 11:19] = np.array([[-mu12, -mu13, 0, 0, 0, 0, 0, 0],
                                                      [0, 0, -mu14, -mu15, 0, 0, 0, 0],
                                                      [+mu12, 0, +mu14, 0, -mu16, -mu17, 0, 0],
                                                      [0, +mu13, 0, 0, +mu16, 0, -mu18, 0],
                                                      [0, 0, 0, +mu15, 0, +mu17, 0, -mu19],
                                                      [0, 0, 0, 0, 0, 0, -mu18, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, -mu19]])
    buffer_processing_matrix[7, 19] = -mu20
    buffer_processing_matrix[8, 20] = -mu21
    buffer_processing_matrix[9, 21] = -mu22
    buffer_processing_matrix[2, 19:22] = (mu20, mu21, mu22)
    constituency_matrix = np.eye(22)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])]
    model_type = 'push-pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(2, 0): (0, 0), (3, 1): (0, 1),
                                    (2, 2): (1, 2), (4, 3): (1, 3),
                                    (3, 4): (2, 4), (4, 5): (2, 5),
                                    (9, 8): (0, 8), (9, 9): (1, 9),
                                    (9, 10): (2, 10),
                                    (9, 11): (7, 11), (10, 12): (7, 12),
                                    (9, 13): (8, 13), (11, 14): (8, 14),
                                    (10, 15): (9, 15), (11, 16): (9, 16),
                                    (2, 19): (7, 19), (2, 20): (8, 20),
                                    (2, 21): (9, 21)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


# Pull only examples with single resource per buffer -----------------------------------------------

def one_warehouse(d=9,
                  mu1=10, mu2=100, mu3=100, mu4=100,
                  cost_per_buffer=np.ones((3, 1)),
                  initial_state=np.ones((3, 1)),
                  capacity=np.ones((3, 1)) * np.inf,
                  job_conservation_flag=False,
                  job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    """
    :param d: mean rate of the demand in the deficit buffer.
    :param mu1: mean rate of the processing resource (activity 1).
    :param mu2: mean rate of the demand resource (activity 2).
    :param mu3: mean rate from retailer to plant/warehouse (buffer 1).
    :param mu4: mean rate from retailer to manufactorer (buffer 2)
    :param cost_per_buffer: cost per buffer.
    :param initial_state: initial state.
    :param capacity: buffer capacity constraints.
    :param job_conservation_flag: forces not to go below zero or above capacity constraints.
    :param job_gen_seed: seed for random job generator.
    """
    # Problem data.
    demand_rate = np.array([0, 0, d])[:, None]
    ind_surplus_buffers = [1]
    buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0],
                                         [+mu1, -mu2, 0, +mu4],
                                         [0, -mu2, 0, 0]])
    constituency_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 0, 0]]),
                                         np.array([[0, 0, 0]]),
                                         np.array([[0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env


def two_warehouses(d1=0.2, d2=0.2,
                   mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1,
                   mu5=0.5, mu6=0.1, mu7=0.1, mu8=0.1,
                   mu9=0.1, mu10=0.1, mu11=0.1, mu12=0.1,
                   cost_per_buffer=np.ones((6, 1)),
                   initial_state=np.ones((6, 1)),
                   capacity=np.ones((6, 1)) * np.inf,
                   job_conservation_flag=True,
                   job_gen_seed: Optional[int] = None,
                   r_to_w_rebalance: Optional[bool] = True,
                   w_to_w_rebalance: Optional[bool] = True) \
        -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, d1, 0, 0, d2])[:, None]
    ind_surplus_buffers = [1, 4]
    if r_to_w_rebalance and w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, -mu5, 0, 0, 0, 0, 0, +mu11, +mu12],
                                             [+mu1, -mu2, 0, +mu4, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, +mu5, +mu6, -mu7, 0, +mu9, 0, -mu11, 0],
                                             [0, 0, 0, 0, 0, 0, +mu7, -mu8, 0, +mu10, 0, 0],
                                             [0, 0, 0, 0, 0, 0, 0, -mu8, 0, 0, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]])
    elif not r_to_w_rebalance and w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, -mu5, 0, 0, 0, 0, +mu11],
                                             [+mu1, -mu2, 0, +mu4, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, +mu5, -mu7, 0, +mu9, 0, -mu11],
                                             [0, 0, 0, 0, 0, +mu7, -mu8, 0, +mu10, 0],
                                             [0, 0, 0, 0, 0, 0, -mu8, 0, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0]])
    elif r_to_w_rebalance and not w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, 0, 0, 0, 0, 0, +mu12],
                                             [+mu1, -mu2, 0, +mu4, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, +mu6, -mu7, 0, +mu9, 0, 0],
                                             [0, 0, 0, 0, 0, +mu7, -mu8, 0, +mu10, 0],
                                             [0, 0, 0, 0, 0, 0, -mu8, 0, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
    elif not r_to_w_rebalance and not w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, 0, 0, 0, 0],
                                             [+mu1, -mu2, 0, +mu4, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, -mu7, 0, +mu9, 0],
                                             [0, 0, 0, 0, +mu7, -mu8, 0, +mu10],
                                             [0, 0, 0, 0, 0, -mu8, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 1]])

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    if not r_to_w_rebalance and not w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0),
                                        (4, 4): (3, 4)}
    elif r_to_w_rebalance and not w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0),
                                        (4, 5): (3, 5)}
    elif not r_to_w_rebalance and w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 4): (0, 4),
                                        (4, 5): (3, 5), (0, 9): (3, 9)}
    else:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 4): (0, 4),
                                        (4, 6): (3, 6), (0, 10): (3, 10)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env


def three_warehouses(d1=0.2, d2=0.2, d3=0.2,
                     mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1, mu5=0.5, mu6=0.1, mu7=0.1, mu8=0.1,
                     mu9=0.1, mu10=0.1, mu11=0.1, mu12=0.1, mu13=0.1, mu14=0.1, mu15=0.1, mu16=0.1,
                     mu17=0.1, mu18=0.1, mu19=0.1, mu20=0.1, mu21=0.1,
                     cost_per_buffer=np.ones((9, 1)),
                     initial_state=np.ones((9, 1)),
                     capacity=np.ones((9, 1)) * np.inf,
                     job_conservation_flag=True,
                     job_gen_seed: Optional[int] = None) \
        -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, d1, 0, 0, d2, 0, 0, d3])[:, None]
    ind_surplus_buffers = [1, 4, 7]
    buffer_processing_matrix = np.zeros((9, 21))
    buffer_processing_matrix[[0, 1], 0] = (-mu1, +mu1)
    buffer_processing_matrix[[1, 2], 1] = (-mu2, -mu2)
    buffer_processing_matrix[0, 2] = +mu3
    buffer_processing_matrix[1, 3] = +mu4
    buffer_processing_matrix[[0, 3], 4] = (-mu5, +mu5)
    buffer_processing_matrix[3, 5] = +mu6
    buffer_processing_matrix[[3, 4], 6] = (-mu7, +mu7)
    buffer_processing_matrix[[4, 5], 7] = (-mu8, -mu8)
    buffer_processing_matrix[3, 8] = +mu9
    buffer_processing_matrix[4, 9] = +mu10
    buffer_processing_matrix[[3, 6], 10] = (-mu11, +mu11)
    buffer_processing_matrix[6, 11] = +mu12
    buffer_processing_matrix[[6, 7], 12] = (-mu13, +mu13)
    buffer_processing_matrix[[7, 8], 13] = (-mu14, -mu14)
    buffer_processing_matrix[6, 14] = +mu15
    buffer_processing_matrix[7, 15] = +mu16
    buffer_processing_matrix[[6, 0], 16] = (-mu17, +mu17)
    buffer_processing_matrix[0, 17] = +mu18
    buffer_processing_matrix[[0, 6], 18] = (-mu19, +mu19)
    buffer_processing_matrix[[3, 0], 19] = (-mu20, +mu20)
    buffer_processing_matrix[[6, 3], 20] = (-mu21, +mu21)

    constituency_matrix = np.zeros((9, 21))
    constituency_matrix[0, [0, 4, 18]] = 1
    constituency_matrix[1, 1] = 1
    constituency_matrix[2, [2, 3, 5]] = 1
    constituency_matrix[3, [6, 10, 19]] = 1
    constituency_matrix[4, 7] = 1
    constituency_matrix[5, [8, 9, 11]] = 1
    constituency_matrix[6, [12, 16, 20]] = 1
    constituency_matrix[7, 13] = 1
    constituency_matrix[8, [14, 15, 17]] = 1

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (3, 4): (0, 4),
                                    (4, 6): (3, 6), (6, 10): (3, 10),
                                    (7, 12): (6, 12), (0, 16): (6, 16),
                                    (6, 18): (0, 18), (0, 19): (3, 19),
                                    (3, 20): (6, 20)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env


def three_warehouses_two_manufacturers_per_area(
        d1=0.2, d2=0.2, d3=0.2, d4=0.2, d5=0.2, d6=0.2,
        mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1, mu5=0.5,
        mu6=0.1, mu7=0.1, mu8=0.1, mu9=0.1, mu10=0.1,
        mu11=0.1, mu12=0.1, mu13=0.1, mu14=0.1, mu15=0.1,
        mu16=0.1, mu17=0.1, mu18=0.1, mu19=0.1, mu20=0.1,
        mu21=0.1, mu22=0.1, mu23=0.1, mu24=0.1, mu25=0.1,
        mu26=0.1, mu27=0.1, mu28=0.1, mu29=0.1, mu30=0.1,
        cost_per_buffer=np.ones((15, 1)),
        initial_state=np.ones((15, 1)),
        capacity=np.ones((15, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, d1, 0, 0, d2, 0, 0, d3, 0, d4, 0, d5, 0, d6])[:, None]
    ind_surplus_buffers = [1, 4, 7, 9, 11, 13]
    buffer_processing_matrix = np.zeros((15, 30))
    buffer_processing_matrix[[0, 1], 0] = (-mu1, +mu1)
    buffer_processing_matrix[[1, 2], 1] = (-mu2, -mu2)
    buffer_processing_matrix[0, 2] = +mu3
    buffer_processing_matrix[1, 3] = +mu4
    buffer_processing_matrix[[0, 3], 4] = (-mu5, +mu5)
    buffer_processing_matrix[3, 5] = +mu6
    buffer_processing_matrix[[3, 4], 6] = (-mu7, +mu7)
    buffer_processing_matrix[[4, 5], 7] = (-mu8, -mu8)
    buffer_processing_matrix[3, 8] = +mu9
    buffer_processing_matrix[4, 9] = +mu10
    buffer_processing_matrix[[3, 6], 10] = (-mu11, +mu11)
    buffer_processing_matrix[6, 11] = +mu12
    buffer_processing_matrix[[6, 7], 12] = (-mu13, +mu13)
    buffer_processing_matrix[[7, 8], 13] = (-mu14, -mu14)
    buffer_processing_matrix[6, 14] = +mu15
    buffer_processing_matrix[7, 15] = +mu16
    buffer_processing_matrix[[6, 0], 16] = (-mu17, +mu17)
    buffer_processing_matrix[0, 17] = +mu18
    buffer_processing_matrix[[0, 6], 18] = (-mu19, +mu19)
    buffer_processing_matrix[[3, 0], 19] = (-mu20, +mu20)
    buffer_processing_matrix[[6, 3], 20] = (-mu21, +mu21)
    buffer_processing_matrix[[0, 9], 21] = (-mu22, +mu22)
    buffer_processing_matrix[9, 22] = +mu23
    buffer_processing_matrix[[9, 10], 23] = (-mu24, -mu24)
    buffer_processing_matrix[[3, 11], 24] = (-mu25, +mu25)
    buffer_processing_matrix[11, 25] = +mu26
    buffer_processing_matrix[[11, 12], 26] = (-mu27, -mu27)
    buffer_processing_matrix[[6, 13], 27] = (-mu28, +mu28)
    buffer_processing_matrix[13, 28] = +mu29
    buffer_processing_matrix[[13, 14], 29] = (-mu30, -mu30)

    constituency_matrix = np.zeros((12, 30))
    constituency_matrix[0, [0, 4, 18, 21]] = 1
    constituency_matrix[1, 1] = 1
    constituency_matrix[2, [2, 3, 5, 22]] = 1
    constituency_matrix[3, [6, 10, 19, 24]] = 1
    constituency_matrix[4, 7] = 1
    constituency_matrix[5, [8, 9, 11, 25]] = 1
    constituency_matrix[6, [12, 16, 20, 27]] = 1
    constituency_matrix[7, 13] = 1
    constituency_matrix[8, [14, 15, 17, 28]] = 1
    constituency_matrix[9, 23] = 1
    constituency_matrix[10, 26] = 1
    constituency_matrix[11, 29] = 1

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (3, 4): (0, 4),
                                    (4, 6): (3, 6), (6, 10): (3, 10),
                                    (7, 12): (6, 12), (0, 16): (6, 16),
                                    (6, 18): (0, 18), (0, 19): (3, 19),
                                    (3, 20): (6, 20), (9, 21): (0, 21),
                                    (11, 24): (3, 24), (13, 27): (6, 27)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env


# Simplified examples---connection between supply and final product buffer is missing--------------

def two_warehouses_simplified(d1=0.2, d2=0.2,
                              mu1=0.1, mu2=0.1, mu3=0.1,
                              mu5=0.5, mu6=0.1, mu7=0.1, mu8=0.1,
                              mu9=0.1, mu11=0.1, mu12=0.1,
                              cost_per_buffer=np.ones((6, 1)),
                              initial_state=np.ones((6, 1)),
                              capacity=np.ones((6, 1)) * np.inf,
                              job_conservation_flag=False,
                              job_gen_seed: Optional[int] = None,
                              r_to_w_rebalance: Optional[bool] = True,
                              w_to_w_rebalance: Optional[bool] = True
                              ) -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, d1, 0, 0, d2])[:, None]
    ind_surplus_buffers = [1, 4]
    if r_to_w_rebalance and w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, -mu5, 0, 0, 0, 0, +mu11, +mu12],
                                             [+mu1, -mu2, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, +mu5, +mu6, -mu7, 0, +mu9, -mu11, 0],
                                             [0, 0, 0, 0, 0, +mu7, -mu8, 0, 0, 0],
                                             [0, 0, 0, 0, 0, 0, -mu8, 0, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1]])
    elif not r_to_w_rebalance and w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, -mu5, 0, 0, 0, +mu11],
                                             [+mu1, -mu2, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, +mu5, -mu7, 0, +mu9, -mu11],
                                             [0, 0, 0, 0, +mu7, -mu8, 0, 0],
                                             [0, 0, 0, 0, 0, -mu8, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 1, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 1],
                                        [0, 0, 0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 0]])
    elif r_to_w_rebalance and not w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, 0, 0, 0, +mu12],
                                             [+mu1, -mu2, 0, 0, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0, 0, 0],
                                             [0, 0, 0, +mu6, -mu7, 0, +mu9, 0],
                                             [0, 0, 0, 0, +mu7, -mu8, 0, 0],
                                             [0, 0, 0, 0, 0, -mu8, 0, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0, 0, 0],
                                        [0, 0, 1, 1, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 0, 0, 1, 1]])
    elif not r_to_w_rebalance and not w_to_w_rebalance:
        buffer_processing_matrix = np.array([[-mu1, 0, +mu3, 0, 0, 0],
                                             [+mu1, -mu2, 0, 0, 0, 0],
                                             [0, -mu2, 0, 0, 0, 0],
                                             [0, 0, 0, -mu7, 0, +mu9],
                                             [0, 0, 0, +mu7, -mu8, 0],
                                             [0, 0, 0, 0, -mu8, 0]])
        constituency_matrix = np.array([[1, 0, 0, 0, 0, 0],
                                        [0, 1, 0, 0, 0, 0],
                                        [0, 0, 1, 0, 0, 0],
                                        [0, 0, 0, 1, 0, 0],
                                        [0, 0, 0, 0, 1, 0],
                                        [0, 0, 0, 0, 0, 1]])

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    if not r_to_w_rebalance and not w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0),
                                        (4, 3): (3, 3)}
    elif r_to_w_rebalance and not w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0),
                                        (4, 4): (3, 4)}
    elif not r_to_w_rebalance and w_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 3): (0, 3),
                                        (4, 4): (3, 4), (0, 7): (3, 7)}
    else:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 3): (0, 3),
                                        (4, 5): (3, 5), (0, 8): (3, 8)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env


def three_warehouses_simplified(d1=0.2, d2=0.2, d3=0.2,
                                mu1=0.1, mu2=0.1, mu3=0.1,
                                mu5=0.5, mu6=0.1, mu7=0.1, mu8=0.1,
                                mu9=0.1, mu11=0.1, mu12=0.1,
                                mu13=0.1, mu14=0.1, mu15=0.1,
                                mu17=0.1, mu18=0.1, mu19=0.1, mu20=0.1,
                                mu21=0.1,
                                cost_per_buffer=np.ones((9, 1)),
                                initial_state=np.ones((9, 1)),
                                capacity=np.ones((9, 1)) * np.inf,
                                job_conservation_flag=True,
                                job_gen_seed: Optional[int] = None,
                                r_to_w_rebalance: Optional[bool] = True) \
        -> ControlledRandomWalk:
    # Problem data.
    demand_rate = np.array([0, 0, d1, 0, 0, d2, 0, 0, d3])[:, None]
    ind_surplus_buffers = [1, 4, 7]
    if r_to_w_rebalance:
        # create the same matrix but activity by activity
        buffer_processing_matrix = np.zeros((9, 18))
        buffer_processing_matrix[[0, 1], 0] = (-mu1, +mu1)
        buffer_processing_matrix[[1, 2], 1] = (-mu2, -mu2)
        buffer_processing_matrix[0, 2] = +mu3
        # skip mu4
        buffer_processing_matrix[[0, 3], 3] = (-mu5, +mu5)
        buffer_processing_matrix[[3], 4] = +mu6
        buffer_processing_matrix[[3, 4], 5] = (-mu7, +mu7)
        buffer_processing_matrix[[4, 5], 6] = (-mu8, -mu8)
        buffer_processing_matrix[3, 7] = +mu9
        # skip mu10
        buffer_processing_matrix[[3, 6], 8] = (-mu11, +mu11)
        buffer_processing_matrix[6, 9] = +mu12
        buffer_processing_matrix[[6, 7], 10] = (-mu13, +mu13)
        buffer_processing_matrix[[7, 8], 11] = (-mu14, -mu14)
        buffer_processing_matrix[6, 12] = +mu15
        # skip mu16
        buffer_processing_matrix[[0, 6], 13] = (+mu17, -mu17)
        buffer_processing_matrix[0, 14] = +mu18
        buffer_processing_matrix[[0, 6], 15] = (-mu19, +mu19)
        buffer_processing_matrix[[0, 3], 16] = (+mu20, -mu20)
        buffer_processing_matrix[[3, 6], 17] = (+mu21, -mu21)

        constituency_matrix = np.zeros((9, 18))
        constituency_matrix[0, [0, 3, 15]] = 1
        constituency_matrix[1, 1] = 1
        constituency_matrix[2, [2, 4]] = 1
        constituency_matrix[3, [5, 8, 16]] = 1
        constituency_matrix[4, 6] = 1
        constituency_matrix[5, [7, 9]] = 1
        constituency_matrix[6, [10, 13, 17]] = 1
        constituency_matrix[7, 11] = 1
        constituency_matrix[8, [12, 14]] = 1

    elif not r_to_w_rebalance:
        buffer_processing_matrix = np.zeros((9, 15))
        buffer_processing_matrix[[0, 1], 0] = (-mu1, +mu1)
        buffer_processing_matrix[[1, 2], 1] = (-mu2, -mu2)
        buffer_processing_matrix[0, 2] = +mu3
        # skip mu4
        buffer_processing_matrix[[0, 3], 3] = (-mu5, +mu5)
        # skip mu6
        buffer_processing_matrix[[3, 4], 4] = (-mu7, +mu7)
        buffer_processing_matrix[[4, 5], 5] = (-mu8, -mu8)
        buffer_processing_matrix[3, 6] = +mu9
        # skip mu10
        buffer_processing_matrix[[3, 6], 7] = (-mu11, +mu11)
        # skip mu12
        buffer_processing_matrix[[6, 7], 8] = (-mu13, +mu13)
        buffer_processing_matrix[[7, 8], 9] = (-mu14, -mu14)
        buffer_processing_matrix[6, 10] = +mu15
        # skip mu16
        buffer_processing_matrix[[0, 6], 11] = (+mu17, -mu17)
        # skip mu18
        buffer_processing_matrix[[0, 6], 12] = (-mu19, +mu19)
        buffer_processing_matrix[[0, 3], 13] = (+mu20, -mu20)
        buffer_processing_matrix[[3, 6], 14] = (+mu21, -mu21)

        constituency_matrix = np.zeros((9, 15))
        constituency_matrix[0, [0, 3, 12]] = 1
        constituency_matrix[1, 1] = 1
        constituency_matrix[2, 2] = 1
        constituency_matrix[3, [4, 7, 13]] = 1
        constituency_matrix[4, 5] = 1
        constituency_matrix[5, 6] = 1
        constituency_matrix[6, [8, 11, 14]] = 1
        constituency_matrix[7, 9] = 1
        constituency_matrix[8, 10] = 1

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    if r_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 3): (0, 3),
                                        (4, 5): (3, 5), (6, 8): (3, 8),
                                        (7, 10): (6, 10), (0, 13): (6, 13),
                                        (6, 15): (0, 15), (0, 16): (3, 16),
                                        (3, 17): (6, 17)}
    elif not r_to_w_rebalance:
        assert job_generator.routes == {(1, 0): (0, 0), (3, 3): (0, 3),
                                        (4, 4): (3, 4), (6, 7): (3, 7),
                                        (7, 8): (6, 8), (0, 11): (6, 11),
                                        (6, 12): (0, 12), (0, 13): (3, 13),
                                        (3, 14): (6, 14)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers
    )
    return env
