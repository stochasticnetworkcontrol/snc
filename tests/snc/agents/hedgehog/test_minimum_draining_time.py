import numpy as np
from src import snc as mdt
from src.snc import compute_load_workload_matrix
from src.snc.environments import examples


def test_compute_minimal_draining_time_push_model():
    minimal_draining_time_theory = 20.
    initial_state = np.array([[10.], [10.], [10.]])
    env = examples.simple_reentrant_line_model(
        alpha1=9, mu1=22, mu2=10, mu3=22, cost_per_buffer=np.ones((3, 1)),
        initial_state=initial_state, capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
        job_gen_seed=42)

    workload_mat = compute_load_workload_matrix(
        env=env, num_wl_vec=None, load_threshold=None, feasible_tol=1e-10).workload_mat

    w = workload_mat @ initial_state
    load = workload_mat @ env._job_generator.demand_rate

    minimal_draining_time_theory /= env.job_generator.sim_time_interval
    minimal_draining_time_directly_from_workload = \
        mdt.compute_minimal_draining_time_from_workload(
            w=w, load=load)
    minimal_draining_time_from_workload = \
        mdt.compute_minimal_draining_time_computing_workload_from_env(
            initial_state=initial_state, env=env, demand_rate=env.job_generator.demand_rate)
    minimal_draining_time_cvxpy = mdt.compute_minimal_draining_time_from_env_cvxpy(
        initial_state=initial_state, env=env)
    minimal_draining_time_cvxpy_dual = mdt.compute_dual_minimal_draining_time_from_env_cvxpy(
        initial_state=initial_state, env=env)
    minimal_draining_time_scipy = mdt.compute_minimal_draining_time_from_env_scipy(
        initial_state=initial_state, env=env)
    minimal_draining_time_scipy_dual = mdt.compute_dual_minimal_draining_time_scipy(
        initial_state=initial_state, env=env)

    np.testing.assert_almost_equal(minimal_draining_time_theory,
                                   minimal_draining_time_directly_from_workload, decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_theory,
                                   minimal_draining_time_from_workload, decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_theory, minimal_draining_time_cvxpy,
                                   decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_theory, minimal_draining_time_cvxpy_dual,
                                   decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_theory, minimal_draining_time_scipy)
    np.testing.assert_almost_equal(minimal_draining_time_theory, minimal_draining_time_scipy_dual)


def perform_test_compute_minimal_draining_time_demand_model(env, initial_state,
                                                            minimal_draining_time_theory):
    minimal_draining_time_from_workload = \
        mdt.compute_minimal_draining_time_computing_workload_from_env(
            initial_state=initial_state, env=env, demand_rate=env.job_generator.demand_rate)
    minimal_draining_time_cvxpy = mdt.compute_minimal_draining_time_from_env_cvxpy(
        initial_state=initial_state, env=env)
    minimal_draining_time_cvxpy_dual = mdt.compute_dual_minimal_draining_time_from_env_cvxpy(
        initial_state=initial_state, env=env)
    minimal_draining_time_scipy = mdt.compute_minimal_draining_time_from_env_scipy(
        initial_state=initial_state, env=env)
    minimal_draining_time_scipy_dual = mdt.compute_dual_minimal_draining_time_scipy(
        initial_state=initial_state, env=env)

    # Since the single server with demand uses the scaled Bernoulli job generator, we have to scale
    # the minimal draining time accordingly.
    minimal_draining_time_cvxpy *= env.job_generator.sim_time_interval
    minimal_draining_time_cvxpy_dual *= env.job_generator.sim_time_interval
    minimal_draining_time_scipy *= env.job_generator.sim_time_interval
    minimal_draining_time_scipy_dual *= env.job_generator.sim_time_interval
    # For this model (demand) and this initial state, the max over the workload vectors should not
    # work and instead Eq. (7.4) from CTCN book (online version) should be used.
    if minimal_draining_time_theory:
        assert np.abs(minimal_draining_time_theory - minimal_draining_time_from_workload) > 1e-1
        minimal_draining_time_ref = minimal_draining_time_theory
    else:
        minimal_draining_time_ref = minimal_draining_time_scipy_dual
    # The LP methods should still work for demand models.
    np.testing.assert_almost_equal(minimal_draining_time_ref, minimal_draining_time_cvxpy,
                                   decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_ref, minimal_draining_time_cvxpy_dual,
                                   decimal=5)
    np.testing.assert_almost_equal(minimal_draining_time_ref, minimal_draining_time_scipy)
    np.testing.assert_almost_equal(minimal_draining_time_ref, minimal_draining_time_scipy_dual)


def test_compute_minimal_draining_time_demand_model_extra_supply():
    initial_state = np.array([[10.], [0.], [0.]])
    env = examples.single_station_demand_model(
        alpha_d=9, mu=10, mus=1e2, mud=1e2, cost_per_buffer=np.array([1, 0.5, 10])[:, None],
        initial_state=initial_state, capacity=np.ones((3, 1)) * np.inf,
        job_conservation_flag=True, job_gen_seed=42)
    perform_test_compute_minimal_draining_time_demand_model(env, initial_state, None)


def test_compute_minimal_draining_time_demand_model_extra_surplus():
    minimal_draining_time_theory = 10./9.
    initial_state = np.array([[0.], [10.], [0.]])
    env = examples.single_station_demand_model(
        alpha_d=9, mu=10, mus=1e2, mud=1e2, cost_per_buffer=np.array([1, 0.5, 10])[:, None],
        initial_state=initial_state, capacity=np.ones((3, 1)) * np.inf,
        job_conservation_flag=True, job_gen_seed=42)
    perform_test_compute_minimal_draining_time_demand_model(env, initial_state,
                                                            minimal_draining_time_theory)


def test_compute_minimal_draining_time_demand_model_extra_demand():
    alpha_d = 9
    mu = 10
    mus = 1e2
    mud = 1e2
    s0 = 10
    initial_state = np.array([[0.], [0.], [s0]])
    env = examples.single_station_demand_model(
        alpha_d=alpha_d, mu=mu, mus=mus, mud=mud, cost_per_buffer=np.array([1, 0.5, 10])[:, None],
        initial_state=initial_state, capacity=np.ones((3, 1)) * np.inf,
        job_conservation_flag=True, job_gen_seed=42)
    perform_test_compute_minimal_draining_time_demand_model(env, initial_state, None)
