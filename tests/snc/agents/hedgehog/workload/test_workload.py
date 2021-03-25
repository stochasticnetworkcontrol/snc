import numpy as np
import src.snc.environments.examples as envex
from src import snc as wl, snc as snctools, snc as alt_methods_test
import src.snc.simulation.utils.validation_utils as validation_utils
from src.snc import StrategicIdlingParams
from src.snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from src.snc.environments import examples


def test_compute_network_load_and_workload_single_server_queue():
    """Check Sec. 5.1.1, W(t) = mu^{-1} x."""
    env = envex.single_server_queue()
    load, xi, nu, constraints = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    load_bis = alt_methods_test.compute_network_load(env)
    # Compare two different methods of computing the network load
    np.testing.assert_almost_equal(load, load_bis, decimal=6)
    # Compare results with theoretical solution
    np.testing.assert_almost_equal(load, env.job_generator.demand_rate, decimal=6)
    np.testing.assert_almost_equal(xi.value, [[1.]], decimal=6)
    np.testing.assert_almost_equal(nu.value, [[1.]], decimal=6)


class TestComputeNetworkLoadAndWorkload:
    """ Test of the function compute_load_workload_matrix from workload.py for various models.

        We compare the result of the function with theoretical results based on computation by hand.
        To do the computation by hand, if the buffer processing matrix B is invertible, then you
        have to invert it and the workload matrix is equal to -CB^{-1} where C is the constituency
        matrix. """

    def test_single_station_demand_model(self):
        """Example 7.1.2. Single-station demand-driven model."""
        alpha_d = 9
        mu = 10
        mus = 1e3
        mud = 1e2
        env = envex.single_station_demand_model(alpha_d=alpha_d, mu=mu, mus=mus, mud=mud)

        # Compute load, workload and sensitivity vectors sorted by load
        load, workload_mat, nu = wl.compute_load_workload_matrix(env)

        # Theoretical results from computation by hand
        # Since we are using the scaled Bernoulli job generator, we have to scale the rates:
        alpha_d *= env.job_generator.sim_time_interval
        mu *= env.job_generator.sim_time_interval
        mus *= env.job_generator.sim_time_interval
        mud *= env.job_generator.sim_time_interval

        load_theory = np.array([alpha_d/mud,
                                alpha_d/mu])
        workload_theory = np.array([[0., 0., 1./mud],
                                    [0., -1./mu, 1./mu]])
        nu_theory = np.array([[0, 1, 0],
                              [1, 0, 0]])
        # Workload vector corresponding to the supplier is not a draining one:
        #  load_s = alpha_d/mus; xi_s = [-1./mud, -1./mud, 1./mud]; nu_s = [0, 0, 1].

        # Sort the theoretical results by load and keep only the first num_wl_vec
        sort_index = np.argsort(load_theory)[::-1]
        load_theory = load_theory[sort_index]
        workload_theory = workload_theory[sort_index, :]
        nu_theory = nu_theory[sort_index, :]
        # Compare results with theoretical results
        np.testing.assert_almost_equal(load, load_theory, decimal=6)
        np.testing.assert_almost_equal(workload_mat, workload_theory, decimal=6)
        np.testing.assert_almost_equal(nu, nu_theory, decimal=6)

    def test_simple_reentrant_line_model(self):
        """Example 4.2.3, Figure 2.9  from CTCN book."""
        alpha1 = 0.3
        mu1 = 0.67
        mu2 = 0.35
        mu3 = 0.67
        env = envex.simple_reentrant_line_model(alpha1=alpha1, mu1=mu1, mu2=mu2, mu3=mu3)

        # Compute load, workload and sensitivity vectors sorted by load
        load, workload_mat, nu = wl.compute_load_workload_matrix(env)

        # Theoretical results from computation by hand
        load_theory = np.array([alpha1/mu1 + alpha1/mu3,
                                alpha1/mu2])
        workload_theory = np.array([[1./mu1 + 1./mu3, 1./mu3, 1./mu3],
                                    [1./mu2, 1./mu2, 0.]])
        nu_theory = np.array([[1, 0],
                              [0, 1]])
        # Sort the theoretical results by load and keep only the first num_wl_vec
        sort_index = np.argsort(load_theory)[::-1]
        load_theory = load_theory[sort_index]
        workload_theory = workload_theory[sort_index, :]
        nu_theory = nu_theory[sort_index, :]
        # Compare results with theoretical results
        np.testing.assert_almost_equal(load, load_theory, decimal=6)
        np.testing.assert_almost_equal(workload_mat, workload_theory, decimal=6)
        np.testing.assert_almost_equal(nu, nu_theory, decimal=6)

    def test_simple_reentrant_line_model_scaled_rates(self):
        """Example 4.2.3, Figure 2.9  from CTCN book."""
        alpha1 = 1
        mu1 = 2.1
        mu2 = 1.1
        mu3 = mu1
        env = envex.simple_reentrant_line_model(alpha1=alpha1, mu1=mu1, mu2=mu2, mu3=mu3)
        max_mu = mu1
        alpha1 /= (max_mu*(1 + env.job_generator.add_max_rate))
        mu1 /= (max_mu*(1 + env.job_generator.add_max_rate))
        mu2 /= (max_mu*(1 + env.job_generator.add_max_rate))
        mu3 /= (max_mu*(1 + env.job_generator.add_max_rate))

        # Compute load, workload and sensitivity vectors sorted by load
        load, workload_mat, nu = wl.compute_load_workload_matrix(env)

        # Theoretical results from computation by hand
        load_theory = alpha1 * np.array([1/mu1 + 1/mu3, 1/mu2])
        workload_theory = np.array([[1./mu1 + 1./mu3, 1./mu3, 1./mu3],
                                    [1./mu2, 1./mu2, 0.]])
        nu_theory = np.array([[1, 0],
                              [0, 1]])
        # Sort the theoretical results by load and keep only the first num_wl_vec
        sort_index = np.argsort(load_theory)[::-1]
        load_theory = load_theory[sort_index]
        workload_theory = workload_theory[sort_index, :]
        nu_theory = nu_theory[sort_index, :]
        # Compare results with theoretical results
        np.testing.assert_almost_equal(load, load_theory, decimal=6)
        np.testing.assert_almost_equal(workload_mat, workload_theory, decimal=6)
        np.testing.assert_almost_equal(nu, nu_theory, decimal=6)

    def test_simple_reentrant_line_with_demand_model(self):
        """Example 7.5.1. Simple re-entrant line with demand, Figure 7.5  from CTCN book"""
        alpha_d = 0.2
        mu1 = 0.3
        mu2 = 0.25
        mu3 = 0.3
        mus = 1
        mud = 1
        env = envex.simple_reentrant_line_with_demand_model(alpha_d=alpha_d, mu1=mu1, mu2=mu2,
                                                            mu3=mu3, mus=mus, mud=mud)

        # Compute load, workload and sensitivity vectors sorted by load
        load, workload_mat, nu = wl.compute_load_workload_matrix(env)

        # Theoretical results from computation by hand
        load_theory = np.array([alpha_d/mu1 + alpha_d/mu3,
                                alpha_d/mu2,
                                alpha_d/mud])
        workload_theory = np.array([[0., -1./mu1, -1./mu1, -1./mu1 - 1./mu3, 1./mu1 + 1./mu3],
                                    [0., 0., -1./mu2, -1./mu2, 1./mu2],
                                    [0., 0., 0., 0., 1./mud]])
        nu_theory = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, 0]])
        # Workload vector corresponding to the supplier is not a draining one:
        #  load_s = alpha_d/mus; xi_s = [-1./mus, -1./mus, -1./mus, -1./mus, 1./mus];
        #  nu_s = [0, 0, 0, 1].

        # Sort the theoretical results by load and keep only the first num_wl_vec
        sort_index = np.argsort(load_theory)[::-1]
        load_theory = load_theory[sort_index]
        workload_theory = workload_theory[sort_index, :]
        nu_theory = nu_theory[sort_index, :]
        # Compare results with theoretical results
        np.testing.assert_almost_equal(load, load_theory, decimal=6)
        np.testing.assert_almost_equal(workload_mat, workload_theory, decimal=6)
        np.testing.assert_almost_equal(nu, nu_theory, decimal=6)


def test_compute_network_load_and_workload_simple_routing_model():
    """Example 6.2.2. from CTCN online, Example 6.2.4 from printed version."""
    alpha_r = 0.18
    mu1 = 0.13
    mu2 = 0.07
    mu_r = 0.3

    env = envex.simple_routing_model(alpha_r, mu1, mu2, mu_r)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    # Compare the two methods of obtaining the network load, directly and as the highest load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=6)
    # Compare results with theoretical result from CTCN book, for the 4 resources.
    load_theory = np.array([alpha_r / (mu1 + mu2),
                            alpha_r / mu_r,
                            0,
                            0])
    workload_theory = np.vstack((1 / (mu1 + mu2) * np.ones(env.num_buffers),
                                 [0, 0, 1/mu_r],
                                 [0, 1 / mu2, 0],
                                 [1/mu1, 0, 0]))
    nu_theory = np.array([[mu1 / (mu1 + mu2), mu2 / (mu1 + mu2), 0],
                          [0, 0, 1],
                          [0, 1, 0],
                          [1, 0, 0]])

    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(workload, decimals=6, out=workload)
    np.around(workload_theory, decimals=6, out=workload_theory)
    np.around(nu, decimals=6, out=nu)
    np.around(nu_theory, decimals=6, out=nu_theory)
    workload_set = set(map(tuple, workload))
    workload_theory_set = set(map(tuple, workload_theory))
    nu_set = set(map(tuple, nu))
    nu_theory_set = set(map(tuple, nu_theory))

    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    assert workload_set == workload_theory_set
    assert nu_set == nu_theory_set


def test_return_workload_vectors_based_on_medium_load_threshold_simple_routing_model():
    """Example 6.2.2. from CTCN online, Example 6.2.4 from printed version. Returns workload vectors
    that correspond with threshold > 0.6."""
    alpha_r = 0.18
    mu1 = 0.13
    mu2 = 0.07
    mu_r = 0.3

    env = envex.simple_routing_model(alpha_r, mu1, mu2, mu_r)

    # Compute load, workload and sensitivity vectors sorted by load
    load_threshold = 0.6
    load, workload, nu = wl.compute_load_workload_matrix(env, load_threshold=load_threshold)

    # Compare results with theoretical result from CTCN book for load >= value.
    load_theory = np.array([alpha_r / (mu1 + mu2),
                            alpha_r / mu_r])
    workload_theory = np.vstack((1 / (mu1 + mu2) * np.ones(env.num_buffers),
                                 [0, 0, 1/mu_r]))
    nu_theory = np.array([[mu1 / (mu1 + mu2), mu2 / (mu1 + mu2), 0],
                          [0, 0, 1]])

    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    np.testing.assert_almost_equal(workload, workload_theory, decimal=6)
    np.testing.assert_almost_equal(nu, nu_theory, decimal=6)


def test_return_workload_vectors_based_on_high_load_threshold_simple_routing_model():
    """Example 6.2.2. from CTCN online, Example 6.2.4 from printed version. Returns workload vectors
    that correspond with threshold > 0.8."""
    alpha_r = 0.18
    mu1 = 0.13
    mu2 = 0.07
    mu_r = 0.3

    env = envex.simple_routing_model(alpha_r, mu1, mu2, mu_r)

    # Compute load, workload and sensitivity vectors sorted by load
    load_threshold = 0.8
    load, workload, nu = wl.compute_load_workload_matrix(env, load_threshold=load_threshold)

    # Compare results with theoretical result from CTCN book, for the 4 resources.
    load_theory = np.array([alpha_r / (mu1 + mu2)])
    workload_theory = 1 / (mu1 + mu2) * np.ones((1, env.num_buffers))
    nu_theory = np.array([[mu1 / (mu1 + mu2), mu2 / (mu1 + mu2), 0]])

    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    np.testing.assert_almost_equal(workload, workload_theory, decimal=6)
    np.testing.assert_almost_equal(nu, nu_theory, decimal=6)


def test_return_no_workload_vectors_due_on_too_high_load_threshold_simple_routing_model():
    """Example 6.2.2. from CTCN online, Example 6.2.4 from printed version. Returns workload vectors
    that correspond with threshold > 0.99, but there are none."""
    alpha_r = 0.18
    mu1 = 0.13
    mu2 = 0.07
    mu_r = 0.3

    env = envex.simple_routing_model(alpha_r, mu1, mu2, mu_r)

    # Compute load, workload and sensitivity vectors sorted by load
    load_threshold = 0.99
    load, workload, nu = wl.compute_load_workload_matrix(env, load_threshold=load_threshold)

    assert load.size == workload.size == nu.size == 0


def test_compute_network_load_and_workload_klimov_model():
    """Example 4.2.1. Klimov model form CTCN online"""
    alpha1 = .1
    alpha2 = .12
    alpha3 = .13
    alpha4 = .14
    mu1 = 0.6
    mu2 = 0.7
    mu3 = 0.8
    mu4 = 0.9

    env = envex.klimov_model(alpha1, alpha2, alpha3, alpha4, mu1, mu2, mu3, mu4)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=6)
    # Compare results with theoretical result from CTCN book.
    np.testing.assert_almost_equal(load, alpha1/mu1 + alpha2/mu2 + alpha3/mu3 + alpha4/mu4,
                                   decimal=6)
    np.testing.assert_almost_equal(workload, np.array([[1/mu1, 1/mu2, 1/mu3, 1/mu4]]), decimal=6)


def test_compute_network_load_and_workload_simple_reentrant_line_model_only_one_vector_required():
    """Example 4.2.3. Simple re-entrant line from CTCN online."""
    alpha1 = 0.3
    mu1 = 0.67
    mu2 = 0.35
    mu3 = 0.67

    env = envex.simple_reentrant_line_model(alpha1, mu1, mu2, mu3)

    # Compute load, workload and sensitivity vectors sorted by load
    num_wl_vec = 1
    load, workload, nu = wl.compute_load_workload_matrix(env, num_wl_vec=num_wl_vec)
    # Compare results with theoretical result from CTCN book.
    load_theory = alpha1 * (1/mu1 + 1/mu3)
    workload_theory = np.array([[1/mu1 + 1/mu3, 1/mu3, 1/mu3]])
    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    np.testing.assert_almost_equal(workload, workload_theory, decimal=6)


def test_compute_network_load_and_workload_simple_reentrant_line_model():
    """Example 4.2.3. Simple re-entrant line from CTCN online."""
    alpha1 = 0.3
    mu1 = 0.67
    mu2 = 0.35
    mu3 = 0.67

    env = envex.simple_reentrant_line_model(alpha1, mu1, mu2, mu3)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=6)
    # Compare results with theoretical result from CTCN book.
    load_theory = np.array([alpha1 * (1/mu1 + 1/mu3), alpha1/mu2])
    workload_theory = np.vstack(([1/mu1 + 1/mu3, 1/mu3, 1/mu3],
                                 [1/mu2, 1/mu2, 0]))
    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    np.testing.assert_almost_equal(workload, workload_theory, decimal=6)


def test_compute_network_load_and_workload_ksrs_network_model_equal_parameters():
    """Example 4.2.4. KSRS model from CTCN online."""
    alpha1 = 0.4
    alpha3 = 0.4
    mu1 = 0.9
    mu2 = 0.8
    mu3 = 0.9
    mu4 = 0.8

    env = envex.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    workload_theory = np.array([[1 / mu1, 0, 1 / mu4, 1 / mu4], [1 / mu2, 1 / mu2, 1 / mu3, 0]])
    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(workload, decimals=6, out=workload)
    np.around(workload_theory, decimals=6, out=workload_theory)
    workload_set = set(map(tuple, workload))
    workload_theory_set = set(map(tuple, workload_theory))

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    # Compare results with theoretical result from CTCN book.
    np.testing.assert_almost_equal(load, np.array([alpha1/mu1 + alpha3/mu4, alpha1/mu2
                                                   + alpha3/mu3]), decimal=6)
    assert workload_set == workload_theory_set


def test_compute_network_load_and_workload_ksrs_network_model_different_parameters():
    """Trying different parameters for Example 4.2.4. KSRS model from CTCN online."""
    alpha1 = 0.4
    alpha3 = 0.2
    mu1 = 0.7
    mu2 = 0.75
    mu3 = 0.8
    mu4 = 0.8

    env = envex.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    workload_theory = np.array([[1 / mu1, 0, 1 / mu4, 1 / mu4], [1 / mu2, 1 / mu2, 1 / mu3, 0]])
    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(workload, decimals=6, out=workload)
    np.around(workload_theory, decimals=6, out=workload_theory)
    workload_set = set(map(tuple, workload))
    workload_theory_set = set(map(tuple, workload_theory))
    assert workload_set == workload_theory_set

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    # Compare the two methods of obtaining the bottleneck workload.
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=6)


def test_compute_network_load_and_workload_dai_wang_model():
    """Example 4.6.4 and Figure 4.15 from CTCN online ed."""

    alpha1 = 0.2
    mu1 = 0.66
    mu2 = mu1
    mu3 = 0.42
    mu4 = mu3
    mu5 = mu1
    env = envex.dai_wang_model(alpha1=alpha1, mu1=mu1, mu2=mu2, mu3=mu3, mu4=mu4, mu5=mu5)
    load_theory = np.array([2 * alpha1/mu3, 3 * alpha1/mu1])
    workload_theory = np.array([[1/mu3 + 1/mu4, 1/mu3 + 1/mu4, 1/mu3 + 1/mu4, 1/mu4, 0],
                                [1/mu1 + 1/mu2 + 1/mu5, 1/mu2 + 1/mu5, 1/mu5, 1/mu5, 1/mu5]])

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=6)
    # Compare results with theoretical result from CTCN book.
    np.testing.assert_almost_equal(load, load_theory, decimal=6)
    np.testing.assert_almost_equal(workload, workload_theory, decimal=6)


def test_compute_network_load_and_workload_simple_link_constrained_with_route_scheduling_model():
    """We follow Figure 6.7 from CTCN book, but we model the resources as being able to schedule one
    output route at a time, as opposed to Example 6.3.1 where each link is a resource. By using the
    Min-Cut Max-Flow theorem, we obtain alpha_star = 3."""
    alpha1 = 4 / 100
    mu12 = 2 / 100
    mu13 = 10 / 100
    mu25 = 1 / 100
    mu32 = 5 / 100
    mu34 = 2 / 100
    mu35 = 2 / 100
    mu45 = 10 / 100
    mu5 = 100 / 100

    env = envex.simple_link_constrained_with_route_scheduling_model(alpha1, mu12, mu13, mu25, mu32,
                                                                    mu34, mu35, mu45, mu5)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=3)
    np.testing.assert_almost_equal(network_load, load[0], decimal=3)
    # Compare the two methods of obtaining the bottleneck workload.
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=3)
    # Compare results with theoretical result from CTCN book, for the 4 resources.
    alpha_star = 3 / 100
    np.testing.assert_almost_equal(load[0], alpha1/alpha_star, decimal=6)
    np.testing.assert_almost_equal(workload[0, :], np.array([1/alpha_star, 1/alpha_star,
                                                             1/alpha_star, 0, 0]), decimal=6)


def test_compute_network_load_and_workload_simple_link_constrained_model():
    """Example 6.3.1. Simple link-constrained model from CTCN online (Example 6.3.5 from printed
    version). Corresponds with Figure 6.7."""
    alpha1 = 4
    mu12 = 2
    mu13 = 10
    mu25 = 1
    mu32 = 5
    mu34 = 2
    mu35 = 2
    mu45 = 10
    mu5 = 100

    env = envex.simple_link_constrained_model(alpha1, mu12, mu13, mu25, mu32, mu34, mu35, mu45, mu5)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=3)
    np.testing.assert_almost_equal(network_load, load[0], decimal=3)
    # Compare the two methods of obtaining the bottleneck workload.
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=3)
    # Compare results with theoretical result from CTCN book, for the 4 resources.
    alpha_star = 5
    np.testing.assert_almost_equal(load[0], alpha1/alpha_star, decimal=6)
    np.testing.assert_almost_equal(workload[0, :] * env.job_generator.sim_time_interval,
                                   np.array([1/alpha_star, 1/alpha_star, 1/alpha_star, 0, 0]),
                                   decimal=6)
    np.testing.assert_almost_equal(nu[0, :], np.array([0, 0, 1/alpha_star, 0, 2/alpha_star,
                                                       2/alpha_star, 0, 0]), decimal=6)


def test_input_queued_switch_3x3_model():
    """Example 6.5.2. Input-queued switch, Figure 6. 10, Proposition 6.6.2, from CTCN book."""
    alpha11 = 0.11
    alpha12 = 0.12
    alpha13 = 0.13
    alpha21 = 0.21
    alpha22 = 0.22
    alpha23 = 0.23
    alpha31 = 0.31
    alpha32 = 0.32
    alpha33 = 0.33
    demand_rate = np.array([alpha11, alpha12, alpha13, alpha21, alpha22, alpha23, alpha31, alpha32,
                            alpha33])[:, None]
    env = envex.input_queued_switch_3x3_model(demand_rate=demand_rate)

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    # Compare the two methods of obtaining the bottleneck workload.
    np.testing.assert_almost_equal(xi_bottleneck.value, np.reshape(workload[0, :],
                                                                   (env.num_buffers, 1)), decimal=2)
    # Compare results with theoretical result from CTCN book, for the 4 resources.
    demand_rate_mat = demand_rate.reshape((3, 3))
    network_load_theory = np.max((np.max(np.sum(demand_rate_mat, axis=0)),
                                  np.max(np.sum(demand_rate_mat, axis=1))))
    np.testing.assert_almost_equal(load[0], network_load_theory, decimal=6)
    assert snctools.is_approx_binary(workload)


def test_three_station_network_model():
    """Example 5.3.2 from CTCN online (Example 5.3.5 from printed version). Figure 5.2."""
    env = envex.three_station_network_model()

    # Compute load, workload and sensitivity vectors sorted by load
    load, workload, nu = wl.compute_load_workload_matrix(env)
    # Third method of computing the network load:
    network_load_bis = alt_methods_test.compute_network_load(env)
    # Compute network load and associated workload and sensitivity vectors. Useful to validate that
    # we obtain the same results with two different methods.
    network_load, xi_bottleneck, nu_bottleneck, constraints \
        = alt_methods_test.compute_network_load_and_bottleneck_workload(env)

    workload_theory = np.array([[1, 1, 0, 1, 0, 1], [1, 0, 1, 1, 0, 1], [1, 0, 1, 0, 1, 1]])
    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(workload, decimals=6, out=workload)
    np.around(workload_theory, decimals=6, out=workload_theory)
    workload_set = set(map(tuple, workload))
    workload_theory_set = set(map(tuple, workload_theory))

    # Compare the three methods of obtaining the network load.
    np.testing.assert_almost_equal(network_load, network_load_bis, decimal=6)
    np.testing.assert_almost_equal(network_load, load[0], decimal=6)
    # We don't compare the two methods of obtaining the bottleneck workload for zero demand.
    # Compare workload with theoretical result from CTCN book.
    assert workload_set == workload_theory_set


def test_limited_workload_vectors_in_three_station_network_model():
    """Example 5.3.2 from CTCN online (Example 5.3.5 from printed version). Figure 5.2. This test
    specifies that only one workload vector should be returned."""
    env = envex.three_station_network_model()

    # Compute load, workload and sensitivity vectors sorted by load
    num_wl_vec = 1
    load, workload, nu = wl.compute_load_workload_matrix(env, num_wl_vec)

    # Compare results with first vector given by from CTCN book.
    np.testing.assert_almost_equal(workload, np.array([[1, 1, 0, 1, 0, 1]]), decimal=6)

    # Compute load, workload and sensitivity vectors sorted by load
    num_wl_vec = 2
    load, workload, nu = wl.compute_load_workload_matrix(env, num_wl_vec)

    # Compare results with first two vectors given by from CTCN book.
    np.testing.assert_almost_equal(workload, np.array([[1, 1, 0, 1, 0, 1],
                                                       [1, 0, 1, 1, 0, 1]]), decimal=6)


def test_three_workload_vectors_in_complex_demand_driven_model():
    """Example 7.2.3, Figure 7.5, from CTCN."""
    params = {'d1': 19/75, 'd2': 19/75, 'mu1': 13/15, 'mu2': 26/15, 'mu3': 13/15, 'mu4': 26/15,
              'mu5': 1, 'mu6': 2, 'mu7': 1, 'mu8': 2, 'mu9': 1, 'mu10a': 1/3, 'mu10b': 1/3,
              'mu11': 1/2, 'mu12': 1/10, 'mud1': 100, 'mus1': 100, 'mud2': 100, 'mus2': 100}
    scaling_factor = 100
    params = {k: v/scaling_factor for k, v in params.items()}
    env = envex.complex_demand_driven_model(**params)

    # Compute load, workload and sensitivity vectors sorted by load
    num_wl_vec = 4
    load, workload, nu = wl.compute_load_workload_matrix(env, num_wl_vec)

    # Compare results with first two vectors given by from CTCN book.
    workload_book_t = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [-0.58, -0.5, -3.03],
            [-1.15, -2, 0],
            [-1.15, 0, 0],
            [-0.58, 0, 0],
            [-1.73, -0.5, -3.03],
            [-0.58, -0.26, -3.03],
            [-1.15, -1, 0],
            [-0.58, -0.5, 0],
            [-0.58, -0.5, -3.03],
            [-0.58, -0.5, 0],
            [-2.02, -1.75, -3.79],
            [-1.73, -2, 0],
            [2.02, 1.75, 3.79],  # Book says [2.01, 1.75, 3.79]
            [1.73, 2, 0]
        ]
    )
    workload_book_t *= scaling_factor

    # network_load, xi_bottleneck, nu_bottleneck, constraints \
    #     = alt_methods_test.compute_network_load_and_bottleneck_workload(env)

    # We obtain 4 bottlenecks. But the compute_vertex routing is susceptible of numerical noise.
    # Check that they all have same network load equal to the one in the book: 0.95.
    np.testing.assert_almost_equal(load, 0.95)

    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(workload, out=workload)

    # We don't get similar but not exactly the same workload vectors as in the book. Most of the
    # values are equal, but there are some that are slightly different. I asked Sean and he said
    # that this example was done by his PhD student several years ago and that he cannot confirm
    # whether the values are correct. So we will compare only with the first workload vector, which
    # only differs from one of the computed workload vectors in the second decimal of one of the
    # values.
    equal = False
    for w in workload:
        equal |= np.allclose(workload_book_t.T[0, :], w)
    assert equal


def test_physical_resources_to_workload_index():
    nu = np.array([[0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]])
    phys_resources_to_wl_index = validation_utils.workload_to_physical_resources_index(nu)
    assert np.all(np.array([[1], [2], [0]]) == phys_resources_to_wl_index)


def test_physical_resources_to_workload_index_no_change():
    nu = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    phys_resources_to_wl_index = validation_utils.workload_to_physical_resources_index(nu)
    assert np.all(np.array([[0], [1], [2]]) == phys_resources_to_wl_index)


def test_physical_resources_to_workload_index_pooled_resources():
    nu = np.array([[0.5, 0, 0.5],
                   [0, 1, 0],
                   [0, 0, 1]])
    phys_resources_to_wl_index = validation_utils.workload_to_physical_resources_index(nu)
    assert np.all(np.array([[0, 2], [1], [2]]) == phys_resources_to_wl_index)


def test_physical_resources_to_workload_index_simple_reentrant_line():
    # For this model and these parameters, the second resource should have higher load and so should
    # correspond to the first workload
    env = examples.simple_reentrant_line_model(
        alpha1=9, mu1=22, mu2=10, mu3=22, cost_per_buffer=np.ones((3, 1)), initial_state=(0, 0, 0),
        capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True, job_gen_seed=42)
    nu = wl.compute_load_workload_matrix(env=env, num_wl_vec=None, load_threshold=None,
                                         feasible_tol=1e-10).nu
    phys_resources_to_wl_index = validation_utils.workload_to_physical_resources_index(nu)
    assert np.all(np.array([[1], [0]]) == phys_resources_to_wl_index)


def test_physical_resources_to_workload_simple_routing_model():
    # For this model, there should be a pooled resource
    env = examples.simple_routing_model(
        alpha_r=0.2, mu1=0.13, mu2=0.07, mu_r=0.2, cost_per_buffer=np.ones((3, 1)),
        initial_state=(1, 1, 1), capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
        job_gen_seed=42)
    nu = wl.compute_load_workload_matrix(env=env, num_wl_vec=None, load_threshold=None,
                                         feasible_tol=1e-10).nu
    phys_resources_to_wl_index = validation_utils.workload_to_physical_resources_index(nu)
    assert np.all(np.array([[2], [0, 1], [1], [0]]) == phys_resources_to_wl_index)


def prepare_describe_workload_space_as_halfspaces_test(num_wl_vec=None):
    env = examples.simple_link_constrained_model(
        mu13=4, alpha1=4.8, mu12=2, mu25=2, mu32=3, mu34=1.7, mu35=2, mu45=1, mu5=5.2,
        cost_per_buffer=np.array([[1], [0.5], [4], [2], [4]]))
    workload_tuple = wl.compute_load_workload_matrix(env, num_wl_vec, use_cdd_for_vertices=True)
    workload_space = wl.describe_workload_space_as_halfspaces(workload_tuple.workload_mat)

    return workload_space, workload_tuple, env


def test_describe_workload_space_as_halfspaces_feasible_point():
    workload_space, workload_tuple, _ = prepare_describe_workload_space_as_halfspaces_test()
    state = np.array([[0], [0], [0.1], [0.1], [0]])
    assert np.all(workload_space @ (workload_tuple.workload_mat @ state) >= -1e-9)


def test_describe_workload_space_as_halfspaces_infeasible_point_negative_buffer():
    workload_space, workload_tuple, _ = prepare_describe_workload_space_as_halfspaces_test()
    state = np.array([[0], [- 0.1], [0.1], [0.1], [0]])
    assert not np.all(workload_space @ (workload_tuple.workload_mat @ state) >= -1e-9)


def test_describe_workload_space_as_halfspaces_infeasible_point_beyond_w_star():
    state = np.array([[0], [0], [0], [0], [1000]])

    workload_space, workload_tuple, env = prepare_describe_workload_space_as_halfspaces_test(2)
    w = workload_tuple.workload_mat @ state

    si_params = StrategicIdlingParams()
    si = StrategicIdlingCore(workload_tuple.workload_mat,
                             workload_tuple.load,
                             env.cost_per_buffer,
                             env.model_type,
                             si_params)
    si._w_param.value = w
    si._w_star_lp_problem.solve(solver="CPLEX", warm_start=True)
    w_star = si._find_workload_with_min_eff_cost_by_idling(w)
    assert np.allclose(w_star, np.array([[1144], [1100]]))
    # w_star is feasible.
    assert np.all(workload_space @ w_star >= -1e-9)
    # Add one to w_star[0] and it becomes infeasible.
    w_beyond_star = np.array([[1145], [1100]])
    assert np.any(workload_space @ w_beyond_star < -1e-9)
