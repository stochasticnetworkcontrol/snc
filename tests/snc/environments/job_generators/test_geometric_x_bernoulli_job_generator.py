import numpy as np

from src.snc.environments \
    import ScaledBernoulliServicesGeometricXBernoulliArrivalsGenerator as ControlDemandVarJobGem
import pytest


def test_geometric_times_bernoulli_job_generator_negative_demand_mean():
    demand_mean = np.array([[-1]])
    demand_var = np.array([[1]])
    with pytest.raises(AssertionError):
        _ = ControlDemandVarJobGem(demand_mean, demand_var, np.array([[-1]]), np.array([[1]]))


def test_geometric_times_bernoulli_job_generator_negative_demand_variance():
    mean = np.array([[1]])
    var = np.array([[-1]])
    with pytest.raises(AssertionError):
        _, _ = ControlDemandVarJobGem(mean, var, np.array([[-1]]), np.array([[1]]))


def test_geometric_times_bernoulli_job_generator_demand_mean_and_var_different_shape():
    mean = np.array([[1], [1]])
    var = np.array([[1]])
    with pytest.raises(AssertionError):
         _, _ = ControlDemandVarJobGem(mean, var, - np.eye(2))


def perform_get_params_test(mean, var, true_geom_p, true_bern_p):
    geom_p, bern_p = ControlDemandVarJobGem.get_params(mean, var)
    np.testing.assert_almost_equal(geom_p, true_geom_p)
    np.testing.assert_almost_equal(bern_p, true_bern_p)


def test_get_params_zero_mean_zero_var():
    mean = np.zeros((4, 1))
    var = np.zeros((4, 1))
    true_geom_p = np.nan * np.ones_like(mean)
    true_bern_p = np.nan * np.ones_like(mean)
    perform_get_params_test(mean, var, true_geom_p, true_bern_p)


def test_get_params_zero_mean_positive_variance():
    mean = np.zeros((4, 1))
    var = np.ones((4, 1))
    true_geom_p = np.zeros_like(mean)
    true_bern_p = np.zeros_like(mean)
    perform_get_params_test(mean, var, true_geom_p, true_bern_p)


def test_get_params_positive_mean_zero_variance():
    mean = np.ones((4, 1))
    var = np.zeros((4, 1))
    true_geom_p = np.ones_like(mean)
    true_bern_p = np.ones_like(mean)
    perform_get_params_test(mean, var, true_geom_p, true_bern_p)


def test_get_params():
    mean = np.array([1, 2, 3, 4])[:, None]
    var = np.array([5, 6, 7, 8])[:, None]
    true_geom_p = np.array([0.2857142857, 0.3333333333, 0.3157894737, 0.2857142857])[:, None]
    true_bern_p = np.array([0.2857142857, 0.6666666667, 0.9473684211, 1.142857143])[:, None]
    perform_get_params_test(mean, var, true_geom_p, true_bern_p)


def test_get_arrival_jobs_zero_mean():
    demand_rate = np.array([[0.2], [0], [0.3], [0]])
    demand_variance = np.array([[0.3], [0], [0.4], [0]])

    np.random.seed(42)
    job_gen_seed = int(40)

    buffer_processing_matrix = - np.eye(4)
    add_max_rate = 0
    job_gen = ControlDemandVarJobGem(demand_rate, demand_variance, buffer_processing_matrix,
                                     job_gen_seed, add_max_rate)
    assert job_gen.sim_time_interval == 1
    num_samples = int(1e5)
    samples = np.zeros((4, num_samples))
    for i in range(num_samples):
        samples[:, [i]] = job_gen.get_arrival_jobs()
    mean_samples = np.mean(samples, axis=1)
    var_samples = np.var(samples, axis=1)
    np.testing.assert_almost_equal(mean_samples, np.squeeze(demand_rate), decimal=2)
    np.testing.assert_almost_equal(var_samples, np.squeeze(demand_variance), decimal=2)


def test_get_arrival_jobs():
    np.random.seed(42)
    job_gen_seed = int(40)

    demand_rate = np.array([.2, .5])[:, None]
    demand_variance = np.array([.3, .9])[:, None]

    buffer_processing_matrix = np.eye(2)
    add_max_rate = 0
    job_gen = ControlDemandVarJobGem(demand_rate, demand_variance, buffer_processing_matrix,
                                     job_gen_seed, add_max_rate)
    assert job_gen.sim_time_interval == 1
    num_samples = int(1e5)
    samples = np.zeros((2, num_samples))
    for i in range(num_samples):
        samples[:, [i]] = job_gen.get_arrival_jobs()
    mean_samples = np.mean(samples, axis=1)
    var_samples = np.var(samples, axis=1)
    np.testing.assert_almost_equal(mean_samples, np.squeeze(demand_rate), decimal=2)
    np.testing.assert_almost_equal(var_samples, np.squeeze(demand_variance), decimal=2)
