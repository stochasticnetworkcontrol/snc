import numpy as np

from snc.agents.hedgehog.asymptotic_workload_cov.\
    compute_asymptotic_cov_bernoulli_service_and_arrivals \
    import ComputeAsymptoticCovBernoulliServiceAndArrivals
from tests.snc.agents.hedgehog.asymptotic_covariance.\
    test_compute_asymptotic_cov_utils import perform_test as perform_test
from snc.environments import examples


def test_compute_covariance_arrival_process():
    demand_rate = np.array([[0.2], [0.4], [0.6]])
    cov_arrival = ComputeAsymptoticCovBernoulliServiceAndArrivals.\
        compute_covariance_arrival_process(demand_rate)
    np.testing.assert_almost_equal(cov_arrival,
                                   np.array([[0.16, 0, 0], [0, 0.24, 0], [0, 0, 0.24]]))


def test_compute_asymptotic_cov_service_process_simple_reentrant_line():
    job_gen_seed = 42
    np.random.seed(job_gen_seed + 100)
    num_batch = 400
    num_data = num_batch * 200
    env = examples.simple_reentrant_line_model(initial_state=np.zeros((3, 1)))
    perform_test(env, num_batch, num_data, ComputeAsymptoticCovBernoulliServiceAndArrivals,
                 job_gen_seed)
