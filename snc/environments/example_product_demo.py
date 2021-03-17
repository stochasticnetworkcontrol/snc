# Set of examples of systems that can be modelled as controlled random walks, from CTCN book.

from typing import Optional
import numpy as np
from snc.environments.job_generators. \
    scaled_bernoulli_services_poisson_arrivals_generator \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
import snc.environments.state_initialiser as stinit
from snc.environments.controlled_random_walk import ControlledRandomWalk


def product_demo_beer_kegs(
        d1=0.006, d2=0.005, d3=0.004,
        d4=0.006, d5=0.005, d6=0.004,
        d7=0.006, d8=0.005, d9=0.004,
        mu1=0.019, mu2=0.019, mu3=0.019,
        mu4=0.016, mu5=0.016, mu6=0.016,
        mu7=0.014, mu8=0.014, mu9=0.014,
        mu10=0.5, mu11=0.5, mu12=0.5,
        mu13=0.5, mu14=0.5, mu15=0.5,
        mu16=0.5, mu17=0.5, mu18=0.5,
        mu19=0.9, mu20=0.9, mu21=0.9,
        cost_per_buffer=np.ones((21, 1)),
        initial_state=np.array(
            [[5], [5], [5], [7], [5], [5], [5], [5], [5], [5], [3], [5], [5], [5], [5], [5], [5],
             [5], [5], [5], [5]]),
        capacity=np.ones((21, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    """
    In this example, we consider the logistic problems associated with beer kegs distribution:
    A depo stores empty kegs and sends them to 3 breweries that produce different beers.
    Each brewery sends full kegs to 3 warehouses located in different non overlapping areas.
    Warehouses have three tasks:
    (1) distribute full kegs to pubs in their area;
    (2) collect empty kegs from the pubs;
    (3) send the empty kegs to the depo.
    In the example, we do not include warehouses' task (1) and (2) and so the warehouses become
    the customers of the network and each of them has 3 demands, one for each type of beer.
    """
    # Problem data.
    demand_rate = np.zeros((21, 1))
    # Three warehouses/customers with (the same) three products each.
    demand_rate[[12, 13, 14, 15, 16, 17, 18, 19, 20], 0] = (d1, d2, d3, d4, d5, d6, d7, d8, d9)

    buffer_processing_matrix = np.zeros((21, 21))
    # Routes from breweries to warehouses, i.e., the customers. There are 3 different products.
    buffer_processing_matrix[[0, 3], 0] = (-mu1, +mu1)
    buffer_processing_matrix[[0, 6], 1] = (-mu2, +mu2)
    buffer_processing_matrix[[0, 9], 2] = (-mu3, +mu3)
    buffer_processing_matrix[[1, 4], 3] = (-mu4, +mu4)
    buffer_processing_matrix[[1, 7], 4] = (-mu5, +mu5)
    buffer_processing_matrix[[1, 10], 5] = (-mu6, +mu6)
    buffer_processing_matrix[[2, 5], 6] = (-mu7, +mu7)
    buffer_processing_matrix[[2, 8], 7] = (-mu8, +mu8)
    buffer_processing_matrix[[2, 11], 8] = (-mu9, +mu9)
    # Processing rates at demand buffers.
    buffer_processing_matrix[[3, 12], 9] = (-mu10, -mu10)
    buffer_processing_matrix[[4, 13], 10] = (-mu11, -mu11)
    buffer_processing_matrix[[5, 14], 11] = (-mu12, -mu12)
    buffer_processing_matrix[[6, 15], 12] = (-mu13, -mu13)
    buffer_processing_matrix[[7, 16], 13] = (-mu14, -mu14)
    buffer_processing_matrix[[8, 17], 14] = (-mu15, -mu15)
    buffer_processing_matrix[[9, 18], 15] = (-mu16, -mu16)
    buffer_processing_matrix[[10, 19], 16] = (-mu17, -mu17)
    buffer_processing_matrix[[11, 20], 17] = (-mu18, -mu18)
    # Supplier, one route per warehouse.
    buffer_processing_matrix[0, 18] = +mu19
    buffer_processing_matrix[1, 19] = +mu20
    buffer_processing_matrix[2, 20] = +mu21

    constituency_matrix = np.zeros((13, 21))
    constituency_matrix[0, [0, 1, 2]] = 1
    constituency_matrix[1, [3, 4, 5]] = 1
    constituency_matrix[2, [6, 7, 8]] = 1
    constituency_matrix[3, 9] = 1
    constituency_matrix[4, 10] = 1
    constituency_matrix[5, 11] = 1
    constituency_matrix[6, 12] = 1
    constituency_matrix[7, 13] = 1
    constituency_matrix[8, 14] = 1
    constituency_matrix[9, 15] = 1
    constituency_matrix[10, 16] = 1
    constituency_matrix[11, 17] = 1
    constituency_matrix[12, [18, 19, 20]] = 1

    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                                    0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        demand_rate, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(3, 0): (0, 0), (6, 1): (0, 1),
                                    (9, 2): (0, 2), (4, 3): (1, 3),
                                    (7, 4): (1, 4), (10, 5): (1, 5),
                                    (5, 6): (2, 6), (8, 7): (2, 7),
                                    (11, 8): (2, 8)}
    state_initialiser = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env
