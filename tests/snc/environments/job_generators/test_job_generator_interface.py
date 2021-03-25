import numpy as np
import pytest
from src.snc.environments \
    import DeterministicDiscreteReviewJobGenerator, PoissonDiscreteReviewJobGenerator
from src.snc.environments \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from src.snc.environments \
    import ScaledBernoulliServicesAndArrivalsGenerator


@pytest.fixture(params=[DeterministicDiscreteReviewJobGenerator,
                        PoissonDiscreteReviewJobGenerator,
                        ScaledBernoulliServicesPoissonArrivalsGenerator,
                        ScaledBernoulliServicesAndArrivalsGenerator])
def class_fixture(request):
    return request.param


def test_create_job_generator_negative_demand(class_fixture):
    demand_rate = np.array([[-0.1]])
    buffer_processing_matrix = np.array([[-3]])
    with pytest.raises(AssertionError):
        _ = class_fixture(demand_rate, buffer_processing_matrix)


def test_create_job_generator_incompatible_size(class_fixture):
    demand_rate = np.ones((2, 1))
    buffer_processing_matrix = np.eye(3)
    with pytest.raises(AssertionError):
        _ = class_fixture(demand_rate, buffer_processing_matrix)


def perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_implementation,
                                                true_routes, true_supply_nodes, true_demand_nodes,
                                                true_exit_nodes):
    job_generator = class_implementation(demand_rate, buffer_processing_matrix)
    routes, supply_nodes, demand_nodes, exit_nodes \
        = job_generator.get_routes_supply_demand_exit_nodes()
    assert routes == true_routes
    assert supply_nodes == true_supply_nodes
    assert demand_nodes == true_demand_nodes
    assert exit_nodes == true_exit_nodes


def test_get_routes_supply_demand_exit_nodes_with_only_exits(class_fixture):
    demand_rate = np.ones((3, 1))
    buffer_processing_matrix = - np.eye(3)
    true_routes = {}
    true_supply_nodes = []
    true_demand_nodes = {}
    true_exit_nodes = [(0, 0), (1, 1), (2, 2)]
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)


def test_get_routes_supply_demand_exit_nodes_with_only_supply(class_fixture):
    demand_rate = np.ones((3, 1))
    buffer_processing_matrix = np.eye(3)
    true_routes = {}
    true_supply_nodes = [(0, 0), (1, 1), (2, 2)]
    true_demand_nodes = {}
    true_exit_nodes = []
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)


def test_get_routes_supply_demand_exit_nodes_with_simple_reentrant_line_with_demand_model(class_fixture):
    demand_rate = np.ones((5, 1))
    mu1 = 1
    mu2 = 2
    mu3 = 3
    mud = 4
    mus = 5
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, mus],
                                         [mu1, -mu2, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0],
                                         [0, 0, mu3, -mud, 0],
                                         [0, 0, 0, -mud, 0]])
    true_routes = {(1, 0): (0, 0),
                   (2, 1): (1, 1),
                   (3, 2): (2, 2)}
    true_supply_nodes = [(0, 4)]
    true_demand_nodes = {(3, 3): (4, 3)}
    true_exit_nodes = []
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)


def test_get_routes_supply_demand_exit_nodes_with_routes_demand_and_multiple_supply(class_fixture):
    demand_rate = np.ones((5, 1))
    mu1 = 1
    mu2 = 2
    mu3 = 3
    mud = 4
    mus1 = 5
    mus2 = 6
    mus3 = 7
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, mus1, mus2, 0],
                                         [mu1, -mu2, 0, 0, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0, 0, 0],
                                         [0, 0, mu3, -mud, 0, 0, mus3],
                                         [0, 0, 0, -mud, 0, 0, 0]])
    true_routes = {(1, 0): (0, 0),
                      (2, 1): (1, 1),
                      (3, 2): (2, 2)}
    true_supply_nodes = [(0, 4), (0, 5), (3, 6)]
    true_demand_nodes = {(3, 3): (4, 3)}
    true_exit_nodes = []
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)


def test_get_routes_supply_demand_exit_nodes_with_routes_multiple_demand_and_multiple_supply(class_fixture):
    demand_rate = np.ones((5, 1))
    mu1 = 1
    mu2 = 2
    mu3 = 3
    mud1 = 4
    mud2 = 5
    mus1 = 6
    mus3 = 7
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, mus1, 0, 0],
                                         [mu1, -mu2, mu3, 0, 0, -mud2, 0],
                                         [0, 0, 0, 0, 0, -mud2, 0],
                                         [0, mu2, -mu3, -mud1, 0, 0, mus3],
                                         [0, 0, 0, -mud1, 0, 0, 0]])
    true_routes = {(1, 0): (0, 0),
                   (3, 1): (1, 1),
                   (1, 2): (3, 2)}
    true_supply_nodes = [(0, 4), (3, 6)]
    true_demand_nodes = {(3, 3): (4, 3), (1, 5): (2, 5)}
    true_exit_nodes = []
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)


def test_get_routes_supply_demand_exit_nodes_with_routes_mul_demand_mul_supply_and_mul_exits(class_fixture):
    demand_rate = np.ones((5, 1))
    mu1 = 1
    mu2 = 2
    mu3 = 3
    mud1 = 4
    mud2 = 5
    mus1 = 6
    mus3 = 7
    mu1e = 8
    mu2e = 9
    buffer_processing_matrix = np.array(
        [[-mu1,    0,    0,     0, mus1,     0,    0, -mu1e,     0],
         [mu1,  -mu2,  mu3,     0,    0, -mud2,    0,     0, -mu2e],
         [0,       0,    0,     0,    0, -mud2,    0,     0,     0],
         [0,     mu2, -mu3, -mud1,    0,     0, mus3,     0,     0],
         [0,       0,    0, -mud1,    0,     0,    0,     0,     0]])
    true_routes = {(1, 0): (0, 0),
                   (3, 1): (1, 1),
                   (1, 2): (3, 2)}
    true_supply_nodes = [(0, 4), (3, 6)]
    true_demand_nodes = {(3, 3): (4, 3), (1, 5): (2, 5)}
    true_exit_nodes = [(0, 7), (1, 8)]
    perform_get_routes_supply_demand_exit_nodes(demand_rate, buffer_processing_matrix,
                                                class_fixture, true_routes, true_supply_nodes,
                                                true_demand_nodes, true_exit_nodes)
