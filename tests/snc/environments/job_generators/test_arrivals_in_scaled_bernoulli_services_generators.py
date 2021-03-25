import numpy as np
import pytest
from snc.environments.job_generators.\
    scaled_bernoulli_services_poisson_arrivals_generator \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from snc.environments.job_generators.\
    scaled_bernoulli_services_and_arrivals_generator \
    import ScaledBernoulliServicesAndArrivalsGenerator


@pytest.fixture(params=[ScaledBernoulliServicesPoissonArrivalsGenerator,
                        ScaledBernoulliServicesAndArrivalsGenerator])
def class_fixture(request):
    return request.param


@pytest.fixture(params=[0, 0.1])
def add_max_rate_fixture(request):
    return request.param


@pytest.fixture(params=[0.9, 5])
def demand_rate_fixture(request):
    demand_rate = np.array([request.param, 0, request.param, 0])[:, None]
    return demand_rate


def test_get_arrival_jobs(demand_rate_fixture, add_max_rate_fixture, class_fixture):
    """Check that the sample average approximates the actual demand mean rate."""
    np.random.seed(42)
    buffer_processing_matrix = np.max(demand_rate_fixture) * 1.1 * np.eye(4)
    job_gen = class_fixture(demand_rate=demand_rate_fixture,
                            buffer_processing_matrix=buffer_processing_matrix,
                            add_max_rate=add_max_rate_fixture)
    num_samples = int(1e5)
    samples = np.zeros((4, 1))
    for _ in range(num_samples):
        new_sample = job_gen.get_arrival_jobs()
        assert np.all(new_sample >= 0)
        samples += new_sample
    max_rate = np.max(np.abs(buffer_processing_matrix))
    mean_samples = (samples / num_samples) * (max_rate * (1 + add_max_rate_fixture))
    np.testing.assert_almost_equal(mean_samples, demand_rate_fixture, decimal=1)
