import numpy as np
import pytest

from src.snc.utils import snc_tools
from src.snc.environments \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from src.snc.environments \
    import ScaledBernoulliServicesAndArrivalsGenerator


@pytest.fixture(params=[ScaledBernoulliServicesPoissonArrivalsGenerator,
                        ScaledBernoulliServicesAndArrivalsGenerator])
def class_fixture(request):
    return request.param


@pytest.fixture(params=[0.1, 0.2, 0.3])
def add_max_rate_fixture(request):
    return request.param


@pytest.fixture(params=[30, -30])
def buffer_processing_matrix_fixture(request):
    buffer_processing_matrix = np.array([[-10, 0, 0, 0],
                                         [10, -3, 0, 0],
                                         [0, 0, request.param, 0],
                                         [0, 0, 20, -3]])
    return buffer_processing_matrix


def test_get_supplied_jobs(class_fixture, add_max_rate_fixture):
    supply_rate = 3
    demand_rate = np.array([2, 0])[:, None]
    # Entry (1, 2) of buffer_processing_matrix is a supply node with mean rate = 3.
    buffer_processing_matrix = np.array([[-10, 0, 0], [0, -4, supply_rate]])
    job_gen = class_fixture(demand_rate=demand_rate,
                            buffer_processing_matrix=buffer_processing_matrix,
                            add_max_rate=add_max_rate_fixture)
    samples = 0
    num_samples = int(1e5)
    for _ in range(num_samples):
        new_sample = job_gen.get_supplied_jobs(job_gen.buffer_processing_matrix[1, 2])
        assert new_sample == 0 or new_sample == 1
        samples += new_sample
    mean_samples = samples / num_samples
    max_rate = np.max(np.abs(buffer_processing_matrix))
    scaled_supply_rate = supply_rate / (max_rate * (1 + add_max_rate_fixture))
    np.testing.assert_almost_equal(mean_samples, scaled_supply_rate, decimal=1)


def test_instantaneous_drained_jobs_matrix(class_fixture):
    demand_rate = np.ones((4, 1))
    buffer_processing_matrix = np.array([[-10, 0, 0, 0],
                                         [10, -3, 0, 0],
                                         [0, 3, -20, 0],
                                         [0, 0, 20, -30]])
    add_max_rate = 0.1  # Default value.
    job_gen = class_fixture(demand_rate=demand_rate,
                            buffer_processing_matrix=buffer_processing_matrix)
    max_rate = np.max(np.abs(buffer_processing_matrix))
    draining_jobs_rate_matrix = np.minimum(0, buffer_processing_matrix) / (
            max_rate * (1 + add_max_rate))
    np.testing.assert_almost_equal(job_gen.draining_jobs_rate_matrix, draining_jobs_rate_matrix)

    samples = np.zeros_like(buffer_processing_matrix)
    num_samples = int(1e5)
    for _ in range(num_samples):
        new_sample = job_gen.get_instantaneous_drained_jobs_matrix()
        assert snc_tools.is_binary_negative(new_sample)
        samples += new_sample
    mean_samples = samples / num_samples
    np.testing.assert_almost_equal(mean_samples, draining_jobs_rate_matrix, decimal=1)

