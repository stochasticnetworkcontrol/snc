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


def perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix, add_max_rate,
                                             true_max_rate, scaled_implementation_class):
    if add_max_rate is not None:
        job_gen = scaled_implementation_class(demand_rate, buffer_processing_matrix,
                                              add_max_rate=add_max_rate)
    else:
        job_gen = scaled_implementation_class(demand_rate, buffer_processing_matrix)
    np.testing.assert_almost_equal(buffer_processing_matrix,
                                   job_gen.unscaled_buffer_processing_matrix)
    np.testing.assert_almost_equal(demand_rate, job_gen.unscaled_demand_rate)
    np.testing.assert_almost_equal(buffer_processing_matrix / true_max_rate,
                                   job_gen.buffer_processing_matrix)
    np.testing.assert_almost_equal(demand_rate / true_max_rate, job_gen.demand_rate)
    np.testing.assert_almost_equal(job_gen.sim_time_interval, 1 / true_max_rate)
    return job_gen


def test_create_scaled_job_generator_large_rates_default_add_max_rate(class_fixture):
    ScaledBernoulliServicesPoissonArrivalsGenerator
    demand_rate = np.array([[3], [2], [1]])
    buffer_processing_matrix = np.array([[-4, 0, 0],
                                         [4, -3, 0],
                                         [0, 3, -2]])
    add_max_rate = None
    true_max_rate = 4.4  # add_max_rate = 0.1 by default.
    perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix, add_max_rate,
                                             true_max_rate, class_fixture)


def test_create_scaled_job_generator_large_rates_zero_add_max_rate(class_fixture):
    demand_rate = np.array([[3], [2], [1]])
    buffer_processing_matrix = np.array([[-4, 0, 0],
                                         [4, -3, 0],
                                         [0, 3, -2]])
    add_max_rate = 0
    true_max_rate = 4
    perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix, add_max_rate,
                                             true_max_rate, class_fixture)


def test_create_scaled_job_generator_large_rates_negative_add_max_rate(class_fixture):
    demand_rate = np.array([[3], [2], [1]])
    buffer_processing_matrix = np.array([[-4, 0, 0],
                                         [4, -3, 0],
                                         [0, 3, -2]])
    add_max_rate = 0
    true_max_rate = -.1
    with pytest.raises(AssertionError):
        perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix,
                                                 add_max_rate, true_max_rate, class_fixture)


def test_create_scaled_job_generator_large_rates_over_unit_add_max_rate(class_fixture):
    demand_rate = np.array([[3], [2], [1]])
    buffer_processing_matrix = np.array([[-4, 0, 0],
                                         [4, -3, 0],
                                         [0, 3, -2]])
    add_max_rate = 0
    true_max_rate = 1.1
    with pytest.raises(AssertionError):
        perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix,
                                                 add_max_rate, true_max_rate, class_fixture)


def test_create_scaled_job_generator_small_rates_default_add_max_rate(class_fixture):
    demand_rate = np.array([[3], [2], [1]])
    buffer_processing_matrix = np.array([[-1, 0, 0],
                                         [1, -0.9, 0],
                                         [0, 0.9, -0.8]])
    add_max_rate = None
    true_max_rate = 1
    job_gen = perform_create_scaled_job_generator_test(demand_rate, buffer_processing_matrix,
                                                       add_max_rate, true_max_rate, class_fixture)
    # In addition to the previous tests, in this case the scaled shouldn't be scaled, so the
    # original rates should be equal to the scaled ones.
    np.testing.assert_almost_equal(buffer_processing_matrix,
                                   job_gen.buffer_processing_matrix)
    np.testing.assert_almost_equal(demand_rate, job_gen.demand_rate)
