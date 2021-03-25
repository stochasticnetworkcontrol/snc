import numpy as np

from src.snc.agents.hedgehog.asymptotic_workload_cov.compute_asymptotic_cov_bernoulli_service_poisson_arrivals \
    import ComputeAsymptoticCovBernoulliServicePoissonArrivals
from .test_compute_asymptotic_cov_utils import perform_test as perform_test
from src.snc.environments import examples


def test_compute_covariance_arrival_process():
    demand_rate = np.array([[1], [2], [3]])
    cov_arrival = ComputeAsymptoticCovBernoulliServicePoissonArrivals.\
        compute_covariance_arrival_process(demand_rate)
    np.testing.assert_almost_equal(cov_arrival, np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]]))


def test_compute_asymptotic_cov_service_process_double_reentrant_line():
    job_gen_seed = 42
    np.random.seed(job_gen_seed + 100)
    num_batch = 400
    num_data = num_batch * 200
    env = examples.double_reentrant_line_only_shared_resources_model(initial_state=np.zeros((4, 1)))
    perform_test(env, num_batch, num_data, ComputeAsymptoticCovBernoulliServicePoissonArrivals,
                 job_gen_seed)
