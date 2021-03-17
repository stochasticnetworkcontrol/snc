import numpy as np
import pytest

import snc.agents.hedgehog.strategic_idling.strategic_idling_utils
from snc.agents.hedgehog.asymptotic_workload_cov.\
    compute_asymptotic_cov_bernoulli_service_and_arrivals \
    import ComputeAsymptoticCovBernoulliServiceAndArrivals
import snc.agents.hedgehog.strategic_idling.hedging_utils as hedging_utils
import snc.agents.hedgehog.workload.workload as wl
from snc.agents.hedgehog.params import StrategicIdlingParams
from snc.agents.hedgehog.strategic_idling.strategic_idling import StrategicIdlingCore
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedgehog_gto import \
    StrategicIdlingGTO, StrategicIdlingHedgehogGTO
from snc.agents.hedgehog.strategic_idling.strategic_idling_hedging import StrategicIdlingHedging
from snc.agents.hedgehog.strategic_idling.strategic_idling_utils import get_dynamic_bottlenecks
import snc.environments.examples as examples
import snc.utils.alt_methods_test as alt_methods_test
import snc.utils.exceptions as exceptions


def test_create_strategic_idling_get_dynamic_bottlenecks():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.69, mu2=0.35, mu3=0.69,
                                               cost_per_buffer=np.array([1, 1, 1])[:, None])
    num_wl_vec = 2
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()

    gto_object = StrategicIdlingGTO(workload_mat=workload_mat,
                                    load=load,
                                    cost_per_buffer=env.cost_per_buffer,
                                    model_type=env.model_type,
                                    strategic_idling_params=strategic_idling_params)

    x = np.array([[158], [856], [0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([1])
    assert set(gto_object.get_allowed_idling_directions(x).k_idling_set) == set([0])

    x = np.array([[493], [476], [0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0,1])
    assert set(gto_object.get_allowed_idling_directions(x).k_idling_set) == set([])

    x = np.array([[631], [338], [0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0])
    assert set(gto_object.get_allowed_idling_directions(x).k_idling_set) == set([1])


def test_create_strategic_idling_hedgehog_gto_normal_hedging():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.69, mu2=0.35, mu3=0.69,
                                               cost_per_buffer=np.array([1.5, 1, 2])[:, None])
    num_wl_vec = 2
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    hgto_object = StrategicIdlingHedgehogGTO(workload_mat=workload_mat,
                                             neg_log_discount_factor=neg_log_discount_factor,
                                             load=load,
                                             cost_per_buffer=env.cost_per_buffer,
                                             model_type=env.model_type,
                                             strategic_idling_params=strategic_idling_params,
                                             workload_cov=workload_cov)

    # this case corresponds to normal hedging regime below hedging threshold
    x = np.array([[631], [338], [0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])

    # this case corresponds to normal hedging regime above hedging threshold
    x = np.array([[969],
                  [  0],
                  [351]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([1])

    # this case corresponds to monotone region
    x = np.array([[493],
                  [476],
                  [  0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0,1])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])

    # this case corresponds to monotone region
    x = np.array([[100],
                  [476],
                  [  0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([1])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])

    assert hgto_object._min_drain_lp is None


def test_create_strategic_idling_hedgehog_gto_switching_curve():
    neg_log_discount_factor = - np.log(0.99999)
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.7, mu2=0.345, mu3=0.7,
                                               cost_per_buffer=np.array([1.5, 1, 2])[:, None])
    num_wl_vec = 2
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    workload_cov = np.array([[2, 0.5], [0.5, 3]])

    h_object = StrategicIdlingHedging(workload_mat=workload_mat,
                                      neg_log_discount_factor=neg_log_discount_factor,
                                      load=load,
                                      cost_per_buffer=env.cost_per_buffer,
                                      model_type=env.model_type,
                                      strategic_idling_params=strategic_idling_params,
                                      workload_cov=workload_cov)

    hgto_object = StrategicIdlingHedgehogGTO(workload_mat=workload_mat,
                                             neg_log_discount_factor=neg_log_discount_factor,
                                             load=load,
                                             cost_per_buffer=env.cost_per_buffer,
                                             model_type=env.model_type,
                                             strategic_idling_params=strategic_idling_params,
                                             workload_cov=workload_cov)

    # This case corresponds to switching curve regime, i.e. minimum cost
    # effective state can only be reached by extending the minimum draining time.
    # `w` is below the hedging threshold so standard Hedgehog would allow one
    # resource to idle, but it turns out that this resource is a dynamic
    # bottleneck for the current `w`.
    x = np.array(([[955],
                   [  0],
                   [202]]))
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])
    assert set(h_object.get_allowed_idling_directions(x).k_idling_set) == set([0])

    # This case corresponds to switching curve regime (i.e., drift @ psi_plus < 0),
    # `w` is below the hedging threshold so standard Hedgehog would allow one resource to idle.
    # Since this resource is not a dynamic bottleneck the GTO constraint also allows it to idle.
    x = np.array([[ 955],
                  [   0],
                  [1112]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([1])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([0])
    assert set(h_object.get_allowed_idling_directions(x).k_idling_set) == set([0])

    # This case corresponds to switching curve regime (i.e., drift @ psi_plus < 0),
    # `w` is below the hedging threshold so standard Hedgehog would allow the
    # less loaded resource to idle. This is similar to the first case, but when both
    # resources are dynamic bottlenecks for the current `w`.
    x = np.array([[759],
                  [  0],
                  [595]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0,1])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])
    assert set(h_object.get_allowed_idling_directions(x).k_idling_set) == set([0])

    # this case corresponds to monotone region so both bottlenecks are not
    # allowed to idle under both standard Hedgehog and GTO policy
    x = np.array([[283],
                  [672],
                  [  0]])
    w = workload_mat @ x
    assert get_dynamic_bottlenecks(w, workload_mat, load) == set([0])
    assert set(hgto_object.get_allowed_idling_directions(x).k_idling_set) == set([])
    assert set(h_object.get_allowed_idling_directions(x).k_idling_set) == set([])

    assert hgto_object._min_drain_lp is not None


def test_create_strategic_idling_no_hedging_object_with_no_asymptotic_covariance():
    """
    Raise exception if asymptotic covariance is tried to be updated.
    """
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.ones((3, 1)))
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()

    x = np.array([[413],
                  [  0],
                  [100]])

    si_object = StrategicIdlingCore(workload_mat=workload_mat, load=load,
                                    cost_per_buffer=env.cost_per_buffer,
                                    model_type=env.model_type,
                                    strategic_idling_params=strategic_idling_params)

    # these methods should not fail
    si_object.get_allowed_idling_directions(x)


def test_create_strategic_idling_object_with_no_asymptotic_covariance():
    """
    Check asymptotic covariance is passed before querying the idling decision
    """
    neg_log_discount_factor = - np.log(0.95)
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.ones((3, 1)))
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()

    x = np.array([[413],
                  [  0],
                  [100]])

    si_object = StrategicIdlingHedging(workload_mat=workload_mat,
                                       neg_log_discount_factor=neg_log_discount_factor, load=load,
                                       cost_per_buffer=env.cost_per_buffer,
                                       model_type=env.model_type,
                                       strategic_idling_params=strategic_idling_params)

    with pytest.raises(AssertionError):
        si_object._verify_offline_preliminaries()

    with pytest.raises(AssertionError):
        si_object.get_allowed_idling_directions(x)


def create_strategic_idling_object(
        workload_mat=np.ones((2, 2)),
        workload_cov=None,
        neg_log_discount_factor=None,
        load=None,
        cost_per_buffer=np.ones((2, 1)),
        model_type='push',
        strategic_idling_params=None):
    if strategic_idling_params is None:
        strategic_idling_params = StrategicIdlingParams()
    return StrategicIdlingHedging(workload_mat=workload_mat,
                                  workload_cov=workload_cov,
                                  neg_log_discount_factor=neg_log_discount_factor,
                                  load=load,
                                  cost_per_buffer=cost_per_buffer,
                                  model_type=model_type,
                                  strategic_idling_params=strategic_idling_params)


def test_create_strategic_idling_object_without_strategic_idling_params():
    """
    Check assert `strategic_idling_params is not None` in constructor.
    """
    neg_log_discount_factor = - np.log(0.95)
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.ones((3, 1)))
    num_wl_vec = 2
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    with pytest.raises(AssertionError):
        _ = StrategicIdlingHedging(workload_mat=workload_mat,
                                   neg_log_discount_factor=neg_log_discount_factor, load=load,
                                   cost_per_buffer=env.cost_per_buffer,
                                   model_type=env.model_type)


def test_is_negative_orthant_true():
    w = np.zeros((3, 1))
    w[0] = -1
    assert StrategicIdlingHedging._is_negative_orthant(w)


def test_is_negative_orthant_false():
    w = np.zeros((3, 1))
    w[0] = 1
    assert not StrategicIdlingHedging._is_negative_orthant(w)


def test_is_negative_orthant_false_since_zero_w():
    w = np.zeros((3, 1))
    assert not StrategicIdlingHedging._is_negative_orthant(w)


def check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer):
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    barc_a, _, eff_cost_a_1 = si_object.c_bar_solver.solve(w)
    _, x_a, eff_cost_a_2 = alt_methods_test.compute_effective_cost_scipy(w, workload_mat,
                                                                         cost_per_buffer)
    barc_b, x_b, eff_cost_b = alt_methods_test.compute_effective_cost_cvxpy(w, workload_mat,
                                                                            cost_per_buffer)
    barc_c, x_c, eff_cost_c = alt_methods_test.compute_dual_effective_cost_cvxpy(w, workload_mat,
                                                                                 cost_per_buffer)
    np.testing.assert_almost_equal(barc_a, barc_b)
    np.testing.assert_almost_equal(barc_a, barc_c)
    np.testing.assert_almost_equal(x_a, x_b)
    np.testing.assert_almost_equal(x_a, x_c)
    np.testing.assert_almost_equal(eff_cost_a_1, eff_cost_b)
    np.testing.assert_almost_equal(eff_cost_a_1, eff_cost_c)
    np.testing.assert_almost_equal(eff_cost_a_1, eff_cost_a_2)
    return barc_a


def test_effective_cost_superfluous_inequalities():
    """We check that Scipy linprog() used in compute_dual_effective_cost() does not return a status
    4 (encountered numerical difficulties)"""
    # This example was known to return this status 4 before the fix
    env = examples.simple_reentrant_line_with_demand_model(alpha_d=2, mu1=3, mu2=2.5, mu3=3,
                                                           mus=1e3, mud=1e3,
                                                           cost_per_buffer=np.ones((5, 1)),
                                                           initial_state=np.array([10, 25,
                                                                                   55, 0,
                                                                                   100])[:, None],
                                                           capacity=np.ones((5, 1)) * np.inf,
                                                           job_conservation_flag=True)
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec=2,
                                                            load_threshold=None)
    w = np.array([[1.], [0.]])
    try:
        si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                                   cost_per_buffer=env.cost_per_buffer,)
        c_bar, _, eff_cost = si_object.c_bar_solver.solve(w)
    except exceptions.ScipyLinprogStatusError:
        pytest.fail()


def test_effective_cost_ksrs_network_model_case_1():
    """Example 5.3.3 case 1 from CTCN book (online version)."""
    mu1 = 1
    mu3 = 1
    mu2 = 1 / 3
    mu4 = 1 / 3
    alpha1 = 0.9
    alpha3 = 0.9
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1 = 1
    w2 = 4
    w = np.array([[w1], [w2]])
    barc_1 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    # Different from CTCN book, [1, 0]
    np.testing.assert_almost_equal(barc_1, 1 / 3 * np.array([[0], [1]]))

    # Region 2 = = {0 < w1 < 3 * w2 < 9 * w1}
    w1 = 2
    w2 = 1
    w = np.array([[w1], [w2]])
    barc_2 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    np.testing.assert_almost_equal(barc_2, 1 / 4 * np.ones((2, 1)))

    # Region 3 = {0 < 3 * w2 < w1}
    w1 = 4
    w2 = 1
    w = np.array([[w1], [w2]])
    barc_3 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    # Different from CTCN book, [1, 0]
    np.testing.assert_almost_equal(barc_3, 1 / 3 * np.array([[1], [0]]))


def test_effective_cost_ksrs_network_model_case_2():
    """Example 5.3.3 case 2 from CTCN book (online version)."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.9
    alpha3 = 0.9
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1 = 1
    w2 = 4
    w = np.array([[w1], [w2]])
    barc_1 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    # Different from CTCN book, [1, -2]
    np.testing.assert_almost_equal(barc_1, np.array([[-2], [1]]))

    # Region 2 = = {0 < w1 < 3 * w2 < 9 * w1}
    w1 = 2
    w2 = 1
    w = np.array([[w1], [w2]])
    barc_2 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    np.testing.assert_almost_equal(barc_2, 1 / 4 * np.ones((2, 1)))

    # Region 3 = {0 < 3 * w2 < w1}
    w1 = 4
    w2 = 1
    w = np.array([[w1], [w2]])
    barc_3 = check_equal_effective_cost_multiple_methods(w, workload_mat, cost_per_buffer)
    # Different from CTCN book, [-2, 1]
    np.testing.assert_almost_equal(barc_3, np.array([[1], [-2]]))


def test_all_effective_cost_vectors_ksrs_network_model_case_1():
    """Example 5.3.3 from CTCN book (online version)."""
    mu1 = 1
    mu3 = 1
    mu2 = 1 / 3
    mu4 = 1 / 3
    alpha1 = 0.9
    alpha3 = 0.9
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Compute cost vectors.
    barc_vectors = alt_methods_test.get_all_effective_cost_linear_vectors(workload_mat,
                                                                          cost_per_buffer)
    barc_vectors_theory = np.array([[1 / 3, 0],
                                    [0, 1 / 3],
                                    [0.25, 0.25]])
    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(barc_vectors, decimals=7, out=barc_vectors)
    np.around(barc_vectors_theory, decimals=7, out=barc_vectors_theory)
    barc_vectors_set = set(map(tuple, barc_vectors))
    barc_vectors_theory_set = set(map(tuple, barc_vectors_theory))
    assert barc_vectors_set == barc_vectors_theory_set


def test_all_effective_cost_vectors_ksrs_network_model_case_2():
    """Example 5.3.3 case 2 from CTCN book (online version)."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.9
    alpha3 = 0.9
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Compute cost vectors.
    barc_vectors = alt_methods_test.get_all_effective_cost_linear_vectors(workload_mat,
                                                                          cost_per_buffer)
    # Order of the vectors not relevant, just made up for easy comparison.
    barc_vectors_theory = np.array([[1, -2],
                                    [-2, 1],
                                    [0.25, 0.25]])
    # Due to numerical noise, different computers can obtain the barc vectors in different order.
    # So we will compare sets instead of ndarrays.
    np.around(barc_vectors, decimals=7, out=barc_vectors)
    np.around(barc_vectors_theory, decimals=7, out=barc_vectors_theory)
    barc_vectors_set = set(map(tuple, barc_vectors))
    barc_vectors_theory_set = set(map(tuple, barc_vectors_theory))
    assert barc_vectors_set == barc_vectors_theory_set


def test_get_vector_defining_possible_idling_direction_1():
    w = np.array([[1], [0]])
    w_star = np.array([[1], [1]])
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    np.testing.assert_almost_equal(v_star, np.array([[0], [1]]))


def test_get_vector_defining_possible_idling_direction_2():
    w = np.array([[0], [1]])
    w_star = np.array([[1], [1]])
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    np.testing.assert_almost_equal(v_star, np.array([[1], [0]]))


def test_get_vector_defining_possible_idling_direction_3():
    # Although this w_star is impossible since w_star >= w, we can still calculate v_star.
    w = np.array([[1], [1]])
    w_star = np.array([[1], [0]])
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    np.testing.assert_almost_equal(v_star, np.array([[0], [-1]]))


def test_get_vector_defining_possible_idling_direction_4():
    # Although this w_star is impossible since w_star >= w, we can still calculate v_star.
    w = np.array([[1], [1]])
    w_star = np.array([[0], [1]])
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)

    np.testing.assert_almost_equal(v_star, np.array([[-1], [0]]))


def test_project_workload_on_monotone_region_along_minimal_cost_negative_w():
    """We use the single server queue with demand model. The expected result when we project
    negative workload with the effective cost LP is zero."""
    env = examples.single_station_demand_model(alpha_d=9, mu=10, mus=1e3, mud=1e2)
    _, workload_mat, _ = wl.compute_load_workload_matrix(env)
    num_wl = workload_mat.shape[0]
    w = - np.ones((num_wl, 1))
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    np.testing.assert_almost_equal(w_star, np.zeros((num_wl, 1)))


def test_project_workload_on_monotone_region_along_minimal_cost_w_equal_w_star_ksrs_region_2():
    """We use the KSRS model, for which we know the boundary of the monotone region. Therefore, if
    we set w in the boundary, we should get w_star = w."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    workload_mat = np.array([[1 / mu1, 0, 1 / mu4, 1 / mu4], [1 / mu2, 1 / mu2, 1 / mu3, 0]])
    cost_per_buffer = np.ones((4, 1))
    # Region 1 = {0 < 3 * w1 < w2 < inf}, and Region 2 = {0 < w1 < 3 * w2 < 9 * w1}, so w = (1, 3)
    # is already right in the boundary.
    w1 = 1
    w2 = 3
    w = np.array([[w1], [w2]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    np.testing.assert_almost_equal(w, w_star)


def test_project_workload_on_monotone_region_along_minimal_cost_ksrs_region_1():
    """We use the KSRS model, for which we know the boundary of the monotone region."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    workload_mat = np.array([[1 / mu1, 0, 1 / mu4, 1 / mu4], [1 / mu2, 1 / mu2, 1 / mu3, 0]])
    cost_per_buffer = np.ones((4, 1))
    # Region 1 = {0 < 3 * w1 < w2 < inf}, so w = (0.5, 3) should be projected to w_star = (1, 3)
    w1 = 0.5
    w2 = 3
    w = np.array([[w1], [w2]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    np.testing.assert_almost_equal(w_star, np.array([[1], [3]]))


def test_project_workload_on_monotone_region_along_minimal_cost_ksrs_region_3():
    """We use the KSRS model, for which we know the boundary of the monotone region."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    workload_mat = np.array([[1 / mu1, 0, 1 / mu4, 1 / mu4], [1 / mu2, 1 / mu2, 1 / mu3, 0]])
    cost_per_buffer = np.ones((4, 1))
    # Region 3 = {0 < 3 * w2 < w1}, so w = (3, 0.5) should be projected to w_star = (3, 1)
    w1 = 3
    w2 = 0.5
    w = np.array([[w1], [w2]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    np.testing.assert_almost_equal(w_star, np.array([[3], [1]]))


def test_project_workload_on_monotone_region_along_minimal_cost_pseudorandom_values():
    """Since this uses random values, it could happen that the simplex (SciPy-LinProg) and SCS (CVX)
    solvers give different solutions. This is uncommon, but possible."""
    np.random.seed(42)
    num_buffers = 4
    num_wl = 3
    num_tests = 1e3

    strategic_idling_params = StrategicIdlingParams()

    discrepancy = 0
    for i in range(int(num_tests)):
        w = np.random.random_sample((num_wl, 1))
        cost_per_buffer = np.random.random_sample((num_buffers, 1))
        workload_mat = np.random.random_sample((num_wl, num_buffers))
        si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                                   cost_per_buffer=cost_per_buffer,
                                                   strategic_idling_params=strategic_idling_params)

        w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
        w_star_b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
            w, workload_mat, cost_per_buffer, "revised simplex")

        if not np.allclose(w_star, w_star_b):
            discrepancy += 1
    assert discrepancy < 5


def test_project_workload_when_monotone_region_is_a_ray():
    """We use the simple re-entrant line model."""
    c_1 = 1
    c_2 = 2
    c_3 = 3
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    mu_1 = 2
    mu_2 = 1
    mu_3 = 2
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    c_plus = np.array([[mu_1 * (c_1 - c_2)],
                       [mu_2 * c_2 + (mu_1 * mu_2) / mu_3 * (c_2 - c_1)]])
    c_minus = np.array([[c_3 * mu_3],
                        [mu_2 * c_1 - c_3 * mu_2 * (mu_3 / mu_1 + 1)]])
    psi_plus = c_plus - c_minus
    w = np.array([[1], [0.]])  # Got from x = np.array([[0.9], [0], [0.2]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    w_star_theory = np.array([w[0], - w[0] * psi_plus[0] / psi_plus[1]])
    np.testing.assert_almost_equal(w_star, w_star_theory)


def test_project_workload_when_idling_direction_lies_in_c_plus_level_set_zero_penalty():
    """We use the simple re-entrant line model."""
    c_1 = 2
    c_2 = 1
    c_3 = 2
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    mu_1 = 2
    mu_2 = 1
    mu_3 = 2
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    c_plus = np.array([[mu_1 * (c_1 - c_2)], [mu_2 * (c_2 * (1 + mu_1/mu_3) - c_1 * mu_1 / mu_3)]])
    c_minus = np.array([[mu_3 * c_3], [mu_2 * (c_1 - c_3 * (1 + mu_3/mu_1))]])
    psi_plus = c_plus - c_minus
    w = np.array([[1], [0.]])  # Got from x = np.array([[0.9], [0], [0.2]])
    strategic_idling_params = StrategicIdlingParams(penalty_coeff_w_star=0)
    si_object = create_strategic_idling_object(
        workload_mat=workload_mat, cost_per_buffer=cost_per_buffer,
        strategic_idling_params=strategic_idling_params)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    w_star_theory = np.array([w[0], - w[0] * psi_plus[0] / psi_plus[1]])
    with pytest.raises(AssertionError):
        np.testing.assert_almost_equal(w_star, w_star_theory)


def test_project_workload_when_idling_direction_lies_in_c_plus_level_set():
    """We use the simple re-entrant line model."""
    c_1 = 2
    c_2 = 1
    c_3 = 2
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    mu_1 = 2
    mu_2 = 1
    mu_3 = 2
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    c_plus = np.array([[mu_1 * (c_1 - c_2)], [mu_2 * (c_2 * (1 + mu_1/mu_3) - c_1 * mu_1 / mu_3)]])
    c_minus = np.array([[mu_3 * c_3], [mu_2 * (c_1 - c_3 * (1 + mu_3/mu_1))]])
    psi_plus = c_plus - c_minus
    w = np.array([[1], [0.]])  # Got from x = np.array([[0.9], [0], [0.2]])
    si_object = create_strategic_idling_object(
        workload_mat=workload_mat, cost_per_buffer=cost_per_buffer,
        strategic_idling_params=StrategicIdlingParams(penalty_coeff_w_star=1e-5))
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    w_star_theory = np.array([w[0], - w[0] * psi_plus[0] / psi_plus[1]])
    np.testing.assert_almost_equal(w_star, w_star_theory, decimal=5)


def test_is_w_inside_monotone_region_ksrs_network_model_case_1():
    """Example 5.3.3 case 1 from CTCN book (online version)."""
    mu1 = 1
    mu3 = 1
    mu2 = 1 / 3
    mu4 = 1 / 3
    alpha1 = 0.3
    alpha3 = 0.3
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Since w is already in \W^+ in any of the 3 regions, any increment in w will increase the cost,
    # so w_star should equal w. Thus, v_star should be a vector of nan, in every case.

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1_1 = 1
    w1_2 = 4
    w_1 = np.array([[w1_1], [w1_2]])
    si_object_1 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer)
    w_star_1 = si_object_1._find_workload_with_min_eff_cost_by_idling(w_1)
    c_bar_1 = si_object_1._get_level_set_for_current_workload(w_1)
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w_1, w_star_1, c_bar_1)

    # Region 2 = = {0 < w1 < 3 * w2 < 9 * w1}
    w2_1 = 2
    w2_2 = 1
    w_2 = np.array([[w2_1], [w2_2]])
    si_object_2 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer)
    w_star_2 = si_object_2._find_workload_with_min_eff_cost_by_idling(w_2)
    c_bar_2 = si_object_2._get_level_set_for_current_workload(w_2)
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w_2, w_star_2, c_bar_2)

    # Region 3 = {0 < 3 * w2 < w1}
    w3_1 = 4
    w3_2 = 0.05
    w_3 = np.array([[w3_1], [w3_2]])
    si_object_3 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer)
    w_star_3 = si_object_3._find_workload_with_min_eff_cost_by_idling(w_3)
    c_bar_3 = si_object_3._get_level_set_for_current_workload(w_3)
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w_3, w_star_3, c_bar_3)


def test_closest_face_ksrs_network_model_case_2():
    """Example 5.3.3 case 2 from CTCN book (online version)."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.3
    alpha3 = 0.3
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    strategic_idling_params = StrategicIdlingParams()
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1_1 = 1
    w1_2 = 4
    w_1 = np.array([[w1_1], [w1_2]])
    si_object_1 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_1 = si_object_1._find_workload_with_min_eff_cost_by_idling(w_1)
    w_star_1b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w_1, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_1, w_star_1b)
    v_star_1 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_1, w_1)
    psi_plus_1, c_plus_1, c_minus_1 = si_object_1._get_closest_face_and_level_sets(w_star_1,
                                                                                   v_star_1)
    np.testing.assert_almost_equal(c_minus_1, np.array([[-2], [1]]), decimal=5)
    np.testing.assert_almost_equal(c_plus_1, np.array([[0.25], [0.25]]), decimal=5)

    # Region 2 = = {0 < w1 < 3 * w2 < 9 * w1}
    w2_1 = 2
    w2_2 = 1
    w_2 = np.array([[w2_1], [w2_2]])

    si_object_2 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_2 = si_object_2._find_workload_with_min_eff_cost_by_idling(w_2)
    w_star_2b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w_2, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_2, w_star_2b)
    # Region 2 is in the monotone region W^+
    c_bar_2 = si_object_2._get_level_set_for_current_workload(w_2)
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w_2, w_star_2, c_bar_2)

    # Region 3 = {0 < 3 * w2 < w1}
    w3_1 = 4
    w3_2 = 0.05
    w_3 = np.array([[w3_1], [w3_2]])

    si_object_3 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_3 = si_object_3._find_workload_with_min_eff_cost_by_idling(w_3)
    w_star_3b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w_3, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_3, w_star_3b)
    v_star_3 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_3, w_3)
    psi_plus_3, c_plus_3, c_minus_3 = si_object_3._get_closest_face_and_level_sets(w_star_3,
                                                                                   v_star_3)
    np.testing.assert_almost_equal(c_minus_3, np.array([[1], [-2]]), decimal=5)
    np.testing.assert_almost_equal(c_plus_3, np.array([[0.25], [0.25]]), decimal=5)


def test_is_monotone_region_a_ray_negative_c_plus():
    c_plus = - np.ones((3, 1))
    assert not StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_nonpositive_c_plus():
    c_plus = np.array([[-1], [-1], [0]])
    assert not StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_zero_c_plus():
    c_plus = np.zeros((3, 1))
    assert not StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_positive_c_plus():
    c_plus = np.ones((3, 1))
    assert not StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_c_plus_with_positive_negative_and_zero_components():
    c_plus = np.array([[1], [-1], [0]])
    assert StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_c_plus_with_positive_and_negative_components():
    c_plus = np.array([[1], [-1], [-1]])
    assert StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_a_ray_simple_reentrant_line():
    """We use the simple re-entrant line with parameters that make monotone region to be a ray."""
    w = np.array([[1], [0]])
    env = examples.simple_reentrant_line_model(mu1=2, mu2=1, mu3=2,
                                               cost_per_buffer=np.array([[1], [2], [3]]))
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)
    assert StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)


def test_is_monotone_region_infeasible_with_real_c_plus():
    c_plus = np.array([[1], [-1], [-1]])
    assert not StrategicIdlingHedging._is_infeasible(c_plus)


def test_is_monotone_region_infeasible():
    c_plus = None
    assert StrategicIdlingHedging._is_infeasible(c_plus)


def test_is_w_inside_monotone_region_when_small_tolerance():
    w = np.random.random_sample((3, 1))
    w_star = w + 1e-4
    c_bar = np.ones((3, 1))
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w, w_star, c_bar)


def test_is_w_inside_monotone_region_false():
    w = np.random.random_sample((3, 1))
    w_star = w + 1e-2
    c_bar = np.ones((3, 1))
    assert not StrategicIdlingHedging._is_w_inside_monotone_region(w, w_star, c_bar)


def check_lambda_star(w, c_plus, psi_plus, w_star, test_strong_duality_flag=True):
    lambda_star = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    lambda_star_b = alt_methods_test.get_price_lambda_star_lp_1_cvxpy(w, c_plus, psi_plus)
    lambda_star_c = alt_methods_test.get_price_lambda_star_lp_2_cvxpy(w, c_plus, psi_plus)
    lambda_star_d = alt_methods_test.get_price_lambda_star_lp_scipy(w, c_plus, psi_plus)
    if test_strong_duality_flag:
        lambda_star_a = alt_methods_test.get_price_lambda_star_strong_duality(w, w_star, c_plus,
                                                                              psi_plus)
        np.testing.assert_almost_equal(lambda_star, lambda_star_a, decimal=5)
    if lambda_star_b is not None:  # If primal is not accurately solved with CVX
        np.testing.assert_almost_equal(lambda_star, lambda_star_b, decimal=5)
    if lambda_star_c is not None:
        np.testing.assert_almost_equal(lambda_star, lambda_star_c, decimal=5)
    np.testing.assert_almost_equal(lambda_star, lambda_star_d)
    return lambda_star


def test_get_price_lambda_star_when_c_plus_is_positive():
    """lambda_star depends on the ratio over the positive components of psi_plus."""
    c_plus = np.array([[1], [1]])
    w = np.array([[3], [0.1]])
    psi_plus = np.array([[-.1], [0.5]])
    check_lambda_star(w, c_plus, psi_plus, None, False)


def test_get_price_lambda_star_when_c_plus_is_negative():
    """c_plus should always be nonnegative"""
    c_plus = np.array([[-1], [1]])
    psi_plus = np.array([[-1], [0.5]])
    with pytest.raises(exceptions.ArraySignError) as excinfo:
        _ = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    assert (excinfo.value.array_name == "c_plus" and excinfo.value.all_components and
            excinfo.value.positive and not excinfo.value.strictly)


def test_get_price_lambda_star_when_c_plus_is_zero():
    """c_plus should always have at least one strictly positive component"""
    c_plus = np.array([[0], [0]])
    psi_plus = np.array([[-1], [0.5]])
    with pytest.raises(exceptions.ArraySignError) as excinfo:
        _ = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    assert (excinfo.value.array_name == "c_plus" and not excinfo.value.all_components and
            excinfo.value.positive and excinfo.value.strictly)


def test_get_price_lambda_star_when_c_plus_has_zero_components():
    """lambda_star only depends on the ratio over the positive components of psi_plus."""
    c_plus = np.array([[0], [1]])
    w = np.array([[3], [0.1]])
    psi_plus = np.array([[-.1], [0.5]])
    check_lambda_star(w, c_plus, psi_plus, None, False)


def test_get_price_lambda_star_when_c_plus_has_zero_components_with_positive_psi_plus():
    """lambda_star only depends on the ratio over the positive components of psi_plus."""
    c_plus = np.array([[0], [1]])
    w = np.array([[-3], [0.1]])
    psi_plus = np.array([[0.5], [0.5]])
    check_lambda_star(w, c_plus, psi_plus, None, False)


def test_get_price_lambda_star_when_psi_plus_is_negative():
    c_plus = np.array([[1], [1]])
    psi_plus = - np.ones((2, 1))
    with pytest.raises(exceptions.EmptyArrayError) as excinfo:
        _ = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    assert excinfo.value.array_name == "ratio"


def test_get_price_lambda_star_when_psi_plus_has_zero_and_positive_components():
    c_plus = np.array([[1], [1]])
    psi_plus = np.array([[0], [1]])
    lambda_star = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    assert lambda_star == 1


def test_get_price_lambda_star_when_psi_plus_has_zero_and_negative_components():
    c_plus = np.array([[1], [1]])
    psi_plus = np.array([[0], [-1]])
    with pytest.raises(exceptions.EmptyArrayError) as excinfo:
        _ = StrategicIdlingHedging._get_price_lambda_star(c_plus, psi_plus)
    assert excinfo.value.array_name == "ratio"


def test_get_price_lambda_star_simple_reentrant_line():
    env = examples.simple_reentrant_line_model(alpha1=0.5, mu1=1.1, mu2=1.2, mu3=1.3)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)
    strategic_idling_params = StrategicIdlingParams()

    for i in range(100):
        # Set w such that a path-wise optimal solution starting from w cannot exist (p. 187,
        # CTCN online ed).
        w1 = i + 1
        w2 = load[1] / load[0] * w1 * 0.9
        w = np.array([[w1], [w2]])

        si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                                   cost_per_buffer=env.cost_per_buffer,
                                                   strategic_idling_params=strategic_idling_params)

        w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
        w_star_b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
            w, workload_mat, env.cost_per_buffer, "revised simplex")

        np.testing.assert_almost_equal(w_star, w_star_b, decimal=5)
        v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
        psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)
        check_lambda_star(w, c_plus, psi_plus, w_star)


def test_get_price_lambda_star_when_monotone_region_is_a_ray_other_workload_value_using_new_cplus():
    """We use the simple re-entrant line with parameters that make monotone region to be a ray."""
    state = np.array([[302], [297], [300]])
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.array([[1], [2], [3]]))

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    w = workload_mat @ state  # = np.array([[59.9], [54.59090909]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)
    assert StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)

    # Set near zero epsilon to get same result with all lambda_star methods.
    psi_plus, c_plus \
        = StrategicIdlingHedging._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
            c_minus, w_star, epsilon=1e-10)
    check_lambda_star(w, c_plus, psi_plus, w_star)

    # Positive epsilon makes the strong duality method for lambda_star give different solution.
    psi_plus, c_plus \
        = StrategicIdlingHedging._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
            c_minus, w_star, epsilon=0.01)
    with pytest.raises(AssertionError):
        check_lambda_star(w, c_plus, psi_plus, w_star)


def test_get_price_lambda_star_when_monotone_region_is_a_ray_with_high_epsilon():
    """We use the simple re-entrant line with parameters that make monotone region to be a ray.
    This test shows that if the artificial cone is very wide, w will be inside, so that we should
    not compute lambda_star."""
    state = np.array([[302], [297], [300]])
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.array([[1], [2], [3]]))

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    w = workload_mat @ state  # = np.array([[59.9], [54.59090909]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)
    assert StrategicIdlingHedging._is_monotone_region_a_ray(c_plus)
    # Positive epsilon makes the strong duality method for lambda_star give different solution.
    psi_plus, c_plus \
        = StrategicIdlingHedging._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
            c_minus, w_star, epsilon=0.3)
    assert psi_plus.T @ w >= 0
    with pytest.raises(AssertionError):
        _ = alt_methods_test.get_price_lambda_star_strong_duality(w, w_star, c_plus, psi_plus)
    with pytest.raises(AssertionError):
        _ = alt_methods_test.get_price_lambda_star_lp_1_cvxpy(w, c_plus, psi_plus)
    with pytest.raises(AssertionError):
        _ = alt_methods_test.get_price_lambda_star_lp_2_cvxpy(w, c_plus, psi_plus)
    with pytest.raises(AssertionError):
        _ = alt_methods_test.get_price_lambda_star_lp_scipy(w, c_plus, psi_plus)


def test_get_price_lambda_star_with_infeasible_workload_space():
    """We use the single server queue with demand for which we know that there is always nonempty
    infeasible region."""
    env = examples.single_station_demand_model(alpha_d=9, mu=10, mus=1e3, mud=1e2,
                                               initial_state=np.array(([300, 0, 1000])))
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec=2)
    w = np.array([[100], [10.01]])
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer)
    w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
    v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
    psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)
    assert StrategicIdlingHedging._is_infeasible(c_plus)
    psi_plus, c_plus = \
        StrategicIdlingHedging._get_closest_face_and_level_sets_for_ray_or_feasibility_boundary(
            c_minus, w_star, epsilon=0)
    check_lambda_star(w, c_plus, psi_plus, w_star)


def test_lambda_star_in_ksrs_network_model_case_2():
    """Example 5.3.3 case 2 from CTCN book (online version)."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.3
    alpha3 = 0.3
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    strategic_idling_params = StrategicIdlingParams()

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1 = 1
    w2 = 4
    w = np.array([[w1], [w2]])
    si_object_1 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=env.cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_1 = si_object_1._find_workload_with_min_eff_cost_by_idling(w)
    w_star_1b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_1, w_star_1b)
    v_star_1 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_1, w)
    psi_plus_1, c_plus_1, c_minus_1 = si_object_1._get_closest_face_and_level_sets(w_star_1,
                                                                                   v_star_1)
    check_lambda_star(w, c_plus_1, psi_plus_1, w_star_1)

    # Region 3 = {0 < 3 * w2 < w1}
    w1 = 4
    w2 = 0.05
    w = np.array([[w1], [w2]])

    si_object_3 = create_strategic_idling_object(
        workload_mat=workload_mat, cost_per_buffer=env.cost_per_buffer,
        strategic_idling_params=strategic_idling_params)

    w_star_3 = si_object_3._find_workload_with_min_eff_cost_by_idling(w)
    w_star_3b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_3, w_star_3b)
    v_star_3 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_3, w)
    psi_plus_3, c_plus_3, c_minus_3 = si_object_3._get_closest_face_and_level_sets(w_star_3,
                                                                                   v_star_3)
    check_lambda_star(w, c_plus_3, psi_plus_3, w_star_3)


def test_compute_height_process_case_1():
    psi_plus = -np.ones((3, 1))
    w = np.ones((3, 1))
    height = StrategicIdlingHedging._compute_height_process(psi_plus, w)
    assert height == 3


def test_compute_height_process_case_2():
    psi_plus = np.array([[-1], [-0.4], [-0.3]])
    w = np.array([[0.2], [1], [2]])
    height = StrategicIdlingHedging._compute_height_process(psi_plus, w)
    np.testing.assert_almost_equal(height, 1.2)


def test_get_possible_idling_directions_single_min_no_threshold():
    w = np.array([[-1], [1]])
    psi_plus = np.array([[1], [-0.5]])
    beta_star = 0
    v_star = np.array([[0.25], [0]])
    k_idling_set = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star, psi_plus,
                                                                          v_star)
    assert np.all(k_idling_set == np.array([0]))


def test_get_possible_idling_directions_single_min_with_high_threshold():
    w = np.array([[-1], [1]])
    psi_plus = np.array([[1], [-0.5]])
    beta_star = 1.5  # Right in the boundary
    v_star = np.array([[0.25], [0]])

    k_idling_set = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star, psi_plus,
                                                                          v_star)
    assert k_idling_set.size == 0


def test_get_possible_idling_directions_multiple_min():
    w = np.array([[-1], [1]])
    psi_plus = np.array([[1], [-0.5]])
    beta_star = 0
    v_star = np.array([[0.25], [0.25]])

    k_idling_set = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star, psi_plus,
                                                                          v_star)
    assert np.all(k_idling_set == np.array([0, 1]))


def test_get_possible_idling_directions_very_small_value_below_tolerance():
    eps = 1e-6
    v_star = np.array([[9e-7], [0]])
    w = np.array([[-1], [1]])
    psi_plus = np.array([[1], [-0.5]])
    beta_star = 0
    k_idling_set = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star, psi_plus,
                                                                          v_star, eps)
    assert k_idling_set.size == 0


def test_get_possible_idling_directions_in_ksrs_network_model_case_2():
    """Example 5.3.3 case 2 from CTCN book (online version)."""
    beta_star = 0  # Set to zero to verify directions of v_star
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.3
    alpha3 = 0.3
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    strategic_idling_params = StrategicIdlingParams()

    # Region 1 = {0 < 3 * w1 < w2 < inf}
    w1 = 1
    w2 = 4
    w = np.array([[w1], [w2]])

    si_object_1 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=env.cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_1 = si_object_1._find_workload_with_min_eff_cost_by_idling(w)
    w_star_1b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w, workload_mat, cost_per_buffer, "revised simplex")

    np.testing.assert_almost_equal(w_star_1, w_star_1b)
    v_star_1 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_1, w)
    psi_plus_1, c_plus_1, c_minus_1 = si_object_1._get_closest_face_and_level_sets(w_star_1,
                                                                                   v_star_1)
    k_idling_set_1 = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star,
                                                                            psi_plus_1, v_star_1)
    assert np.all(k_idling_set_1 == np.array([0]))

    # Region 2 = {0 < w1 < 3 * w2 < 9 * w1} ==> Already in the monotone region W^+
    w1_2 = 2
    w2_2 = 1
    w_2 = np.array([[w1_2], [w2_2]])

    si_object_2 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=env.cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_2 = si_object_2._find_workload_with_min_eff_cost_by_idling(w_2)
    w_star_2b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w_2, workload_mat, cost_per_buffer, "revised simplex")
    np.testing.assert_almost_equal(w_star_2, w_star_2b)
    # Region 2 is in the monotone region W^+
    c_bar_2 = si_object_2._get_level_set_for_current_workload(w_2)
    assert StrategicIdlingHedging._is_w_inside_monotone_region(w_2, w_star_2, c_bar_2)

    # Region 3 = {0 < 3 * w2 < w1}
    w1 = 4
    w2 = 0.05
    w = np.array([[w1], [w2]])

    si_object_3 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=env.cost_per_buffer,
                                                 strategic_idling_params=strategic_idling_params)

    w_star_3 = si_object_3._find_workload_with_min_eff_cost_by_idling(w)
    w_star_3b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
        w, workload_mat, cost_per_buffer, "revised simplex")
    np.testing.assert_almost_equal(w_star_3, w_star_3b)
    v_star_3 = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star_3, w)
    psi_plus_3, c_plus_3, c_minus_3 = si_object_3._get_closest_face_and_level_sets(w_star_3,
                                                                                   v_star_3)

    k_idling_set_3 = StrategicIdlingHedging._get_possible_idling_directions(w, beta_star,
                                                                            psi_plus_3, v_star_3)
    assert np.all(k_idling_set_3 == np.array([1]))


def test_get_possible_idling_directions_simple_reentrant_line():
    env = examples.simple_reentrant_line_model(alpha1=0.5, mu1=1.1, mu2=1.2, mu3=1.3)

    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    # Find all c_bar vectors.
    v = alt_methods_test.get_all_effective_cost_linear_vectors(workload_mat, env.cost_per_buffer)

    strategic_idling_params = StrategicIdlingParams()

    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=env.cost_per_buffer,
                                               strategic_idling_params=strategic_idling_params)

    for i in range(100):
        # Set w such that a path-wise optimal solution starting from w cannot exist (p. 187,
        # CTCN online ed).
        w1 = i + 1
        w2 = load[1] / load[0] * w1 * 0.9
        w = np.array([[w1], [w2]])

        w_star = si_object._find_workload_with_min_eff_cost_by_idling(w)
        w_star_b = alt_methods_test.find_workload_with_min_eff_cost_by_idling_scipy(
            w, workload_mat, env.cost_per_buffer, "revised simplex")

        np.testing.assert_almost_equal(w_star, w_star_b, decimal=5)
        v_star = StrategicIdlingHedging._get_vector_defining_possible_idling_direction(w_star, w)
        psi_plus, c_plus, c_minus = si_object._get_closest_face_and_level_sets(w_star, v_star)

        np.testing.assert_almost_equal(np.hstack((c_plus, c_minus)).T, v, decimal=4)


def test_null_strategic_idling_values():
    si_object = create_strategic_idling_object()
    si_tuple = si_object._get_null_strategic_idling_output(w=np.array([[0]]))
    assert si_tuple.beta_star == 0
    assert si_tuple.k_idling_set.size == 0
    assert si_tuple.sigma_2_h == 0
    assert si_tuple.psi_plus is None
    assert si_tuple.w_star is None
    assert si_tuple.c_plus is None
    assert si_tuple.c_bar is None


def test_is_pull_model_false():
    assert not snc.agents.hedgehog.strategic_idling.strategic_idling_utils.is_pull_model('push')


def test_is_pull_model_true():
    assert snc.agents.hedgehog.strategic_idling.strategic_idling_utils.is_pull_model('pull')


def test_get_index_deficit_buffers():
    workload_mat = np.triu(- np.ones((3, 3)))
    workload_mat = np.hstack((workload_mat, np.ones((3, 1))))
    assert hedging_utils.get_index_deficit_buffers(workload_mat) == [3]


def test_get_index_deficit_buffers_3_buffers():
    workload_mat = np.triu(- np.ones((3, 3)))
    workload_mat = np.hstack((workload_mat, np.ones((3, 1)), np.ones((3, 1)), np.ones((3, 1))))
    assert hedging_utils.get_index_deficit_buffers(workload_mat) == [3, 4, 5]


def test_get_index_deficit_buffers_2_buffers_not_consecutive():
    workload_mat = np.triu(- np.ones((3, 3)))
    workload_mat = np.hstack((np.ones((3, 1)), workload_mat, np.ones((3, 1))))
    assert hedging_utils.get_index_deficit_buffers(workload_mat) == [0, 4]


def test_get_index_deficit_buffers_inconsistent_column():
    workload_mat = np.hstack((np.triu(- np.ones((2, 2))), np.array([[-1], [1]]), np.ones((2, 1))))
    with pytest.raises(AssertionError):
        _ = hedging_utils.get_index_deficit_buffers(workload_mat)


def test_build_state_list_for_computing_cone_envelope_0_dim():
    num_buffers = 0
    init_x = 10
    with pytest.raises(AssertionError):
        _ = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(num_buffers, init_x)


def test_build_state_list_for_computing_cone_envelope_null_initx():
    num_buffers = 1
    init_x = 0
    with pytest.raises(AssertionError):
        _ = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(num_buffers, init_x)


def test_build_state_list_for_computing_cone_envelope_1_dim():
    num_buffers = 1
    init_x = 10
    state_list = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(num_buffers,
                                                                                      init_x)
    assert state_list == [[init_x]]


def test_build_state_list_for_computing_cone_envelope_2_dim():
    num_buffers = 2
    init_x = 10.
    state_list = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(num_buffers,
                                                                                      init_x)
    assert len(state_list) == 3
    assert np.any(np.where(np.array([[init_x], [0]]) == state_list))
    assert np.any(np.where(np.array([[0], [init_x]]) == state_list))
    assert np.any(np.where(np.array([[init_x], [init_x]]) == state_list))


def test_build_state_list_for_computing_cone_envelope_3_dim():
    num_buffers = 3
    init_x = 10.

    state_list = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(num_buffers,
                                                                                      init_x)
    assert len(state_list) == 7
    assert np.any(np.where(state_list == np.array([[init_x], [0], [0]])))
    assert np.any(np.where(state_list == np.array([[0], [init_x], [0]])))
    assert np.any(np.where(state_list == np.array([[0], [0], [init_x]])))
    assert np.any(np.where(state_list == np.array([[init_x], [init_x], [0]])))
    assert np.any(np.where(state_list == np.array([[init_x], [0], [init_x]])))
    assert np.any(np.where(state_list == np.array([[0], [init_x], [init_x]])))
    assert np.any(np.where(state_list == np.array([[init_x], [init_x], [init_x]])))


@pytest.mark.parametrize("max_points", [1, 3, 5])
def test_build_state_list_for_computing_cone_envelope_3_dim_max_points(max_points):
    num_buffers = 3
    init_x = 10.
    state_list = StrategicIdlingHedging._build_state_list_for_computing_cone_envelope(
        num_buffers, init_x, max_points)
    assert len(state_list) == max_points


def test_build_workloads_for_computing_cone_envelope_1():
    workload_mat = np.array([[-1, -2, 1], [0, -1, 0.5]])
    w_list = StrategicIdlingHedging._build_workloads_for_computing_cone_envelope(workload_mat)
    assert len(w_list) == 7
    assert np.any(np.where(w_list == np.array([[-10], [0]])))
    assert np.any(np.where(w_list == np.array([[-20], [-10]])))
    assert np.any(np.where(w_list == np.array([[10], [5]])))
    assert np.any(np.where(w_list == np.array([[-30], [-10]])))
    assert np.any(np.where(w_list == np.array([[0], [5]])))
    assert np.any(np.where(w_list == np.array([[-10], [-5]])))
    assert np.any(np.where(w_list == np.array([[-20], [-5]])))


def test_add_face_to_cone_envelope_already_in_cone_1():
    psi_plus = np.ones((2, 1))
    beta_star = 2
    psi_plus_cone_list = [np.ones((2, 1))]
    beta_star_cone_list = [2]
    si_object = create_strategic_idling_object()
    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_object._add_face_to_cone_envelope(psi_plus, beta_star)
    assert len(si_object.psi_plus_cone_list) == 1
    assert np.all(si_object.psi_plus_cone_list[0] == np.ones((2, 1)))
    assert si_object.beta_star_cone_list == [2]


def test_add_face_to_cone_envelope_already_in_2():
    psi_plus = np.ones((2, 1))
    beta_star = 1
    psi_plus_cone_list = [np.ones((2, 1)), 0.1 * np.ones((2, 1))]
    beta_star_cone_list = [1, 2]
    si_object = create_strategic_idling_object()
    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_object._add_face_to_cone_envelope(psi_plus, beta_star)
    assert len(si_object.psi_plus_cone_list) == 2
    assert np.all(si_object.psi_plus_cone_list[0] == np.ones((2, 1)))
    assert np.all(si_object.psi_plus_cone_list[1] == 0.1 * np.ones((2, 1)))
    assert si_object.beta_star_cone_list == [1, 2]


def test_add_face_to_cone_envelope_already_in_but_different_beta_star_keep_largest_one():
    psi_plus = np.ones((2, 1))
    beta_star = 3
    psi_plus_cone_list = [np.ones((2, 1)), 0.1 * np.ones((2, 1))]
    beta_star_cone_list = [1, 2]
    si_object = create_strategic_idling_object()
    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_object._add_face_to_cone_envelope(psi_plus, beta_star)
    assert len(si_object.psi_plus_cone_list) == 2
    assert np.all(si_object.psi_plus_cone_list[0] == np.ones((2, 1)))
    assert np.all(si_object.psi_plus_cone_list[1] == 0.1 * np.ones((2, 1)))
    assert si_object.beta_star_cone_list == [3, 2]


def test_add_face_to_cone_envelope_new_face():
    psi_plus = 3 * np.ones((2, 1))
    beta_star = 3
    psi_plus_cone_list = [np.ones((2, 1)), 2 * np.ones((2, 1))]
    beta_star_cone_list = [1, 2]
    si_object = create_strategic_idling_object()
    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_object._add_face_to_cone_envelope(psi_plus, beta_star)
    assert len(si_object.psi_plus_cone_list) == 3
    assert np.all(si_object.psi_plus_cone_list[0] == np.ones((2, 1)))
    assert np.all(si_object.psi_plus_cone_list[1] == 2 * np.ones((2, 1)))
    assert np.all(si_object.psi_plus_cone_list[2] == 3 * np.ones((2, 1)))
    assert si_object.beta_star_cone_list == [1, 2, 3]


def build_si_object():
    env = examples.simple_reentrant_line_model()
    neg_log_discount_factor = - np.log(0.95)
    workload_cov = 10 * np.eye(2)
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    return StrategicIdlingHedging(workload_mat=workload_mat,
                                  workload_cov=workload_cov,
                                  neg_log_discount_factor=neg_log_discount_factor,
                                  load=load,
                                  cost_per_buffer=env.cost_per_buffer,
                                  model_type=env.model_type,
                                  strategic_idling_params=StrategicIdlingParams())


def test_compute_cone_envelope_zero_max_points():
    max_points = 0
    si_object = build_si_object()
    with pytest.raises(AssertionError):
        si_object._compute_cone_envelope(max_points)


def test_compute_cone_envelope_float_max_points():
    max_points = 1.0
    si_object = build_si_object()
    with pytest.raises(AssertionError):
        si_object._compute_cone_envelope(max_points)


def test_compute_cone_envelope():
    """
    Use cbar vectors computed manually for demand models that allow negative workload.
    """
    # @TODO
    pass


def test_height_process():
    """Simple correctness test."""
    psi_plus = np.array([[1], [0.5]])
    workload_cov = np.array([[2, 0.5], [0.5, 3]])
    load = np.array([0.9, 0.9])
    delta_h, sigma_2_h = StrategicIdlingHedging._get_height_process_statistics(
        psi_plus, workload_cov, load)

    test_delta_h = 0.15
    test_sigma_2_h = 3.25
    np.testing.assert_almost_equal(delta_h, test_delta_h)
    np.testing.assert_almost_equal(sigma_2_h, test_sigma_2_h)


def test_get_c_plus_when_monotone_region_is_a_ray():
    w_star = np.array([[1], [1]])
    c_minus = np.array([[6], [-5]])

    epsilon = 0.1

    # epsilon_naught = c_minus @ w_star / ||w_star||^2 = 0.5
    # c_plus_theory = (epsilon_naught + epsilon / ||w_star||) * w_star
    c_plus_theory = 0.570710678118655 * np.ones((2, 1))

    c_plus = StrategicIdlingHedging._get_c_plus_when_monotone_region_is_a_ray_or_boundary(
        c_minus, w_star, epsilon)
    np.testing.assert_almost_equal(c_plus, c_plus_theory)


def test_compute_hedging_threshold_with_scaling_factor():
    """
    Check that the hedging threshold value returned by the algorithm equals the one obtained with
    the threshold heuristic times parameter 'hedging_scaling_factor'.
    """
    c_1 = 1.5
    c_2 = 1
    c_3 = 2
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    alpha = 1
    mu_1 = 2.06
    mu_2 = 1.05
    mu_3 = 2.06
    discount_factor = 0.999
    env = examples.simple_reentrant_line_model(alpha, mu_1, mu_2, mu_3, cost_per_buffer)
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    load = workload_mat @ env.job_generator.demand_rate

    cas_cov = ComputeAsymptoticCovBernoulliServiceAndArrivals(
        env.job_generator, env.constituency_matrix, workload_mat)
    workload_cov = cas_cov.compute_asymptotic_workload_cov()
    c_plus = np.array(
        [[mu_1 * (c_1 - c_2)], [mu_2 * (c_2 * (1 + mu_1 / mu_3) - c_1 * mu_1 / mu_3)]])
    c_minus = np.array([[mu_3 * c_3], [mu_2 * (c_1 - c_3 * (1 + mu_3 / mu_1))]])
    psi_plus = c_plus - c_minus

    hedging_scaling_factor = 10
    si_object = StrategicIdlingHedging(
        workload_mat=workload_mat,
        cost_per_buffer=cost_per_buffer,
        model_type=env.model_type,
        strategic_idling_params=StrategicIdlingParams(),
        workload_cov=workload_cov,
        neg_log_discount_factor=-np.log(discount_factor),
        load=load)
    beta_star, sigma_2_h, delta_h, lambda_star, theta_roots = \
        si_object._compute_hedging_threshold(c_plus, psi_plus)

    si_object_b = StrategicIdlingHedging(
        workload_mat=workload_mat,
        cost_per_buffer=cost_per_buffer,
        model_type=env.model_type,
        strategic_idling_params=StrategicIdlingParams(
            hedging_scaling_factor=hedging_scaling_factor),
        workload_cov=workload_cov,
        neg_log_discount_factor=-np.log(discount_factor),
        load=load)
    beta_star_b, sigma_2_h_b, delta_h_b, lambda_star_b, theta_roots_b = \
        si_object_b._compute_hedging_threshold(c_plus, psi_plus)
    np.testing.assert_almost_equal(hedging_scaling_factor * beta_star, beta_star_b)
    np.testing.assert_almost_equal(sigma_2_h, sigma_2_h_b)
    np.testing.assert_almost_equal(delta_h, delta_h_b)
    np.testing.assert_almost_equal(lambda_star, lambda_star_b)
    np.testing.assert_almost_equal(theta_roots, theta_roots_b)


def test_hedging_inside_monotone_region_when_it_is_a_proper_cone():
    """Test all steps for hedging with positive workload when the monotone region is a proper cone
    and we are inside the monotone region. We use KSRS case 2, example 5.3.3 from CTCN book
    (online version). Since we are in the monotone region, beta_star must be zero."""
    mu1 = 1 / 3
    mu3 = 1 / 3
    mu2 = 1
    mu4 = 1
    alpha1 = 0.3
    alpha3 = 0.3
    cost_per_buffer = np.ones((4, 1))
    env = examples.ksrs_network_model(alpha1, alpha3, mu1, mu2, mu3, mu4, cost_per_buffer)
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)
    # Region 2 = {0 < w1 < 3 * w2 < 9 * w1} ==> Already in the monotone region W^+
    w1 = 2
    w2 = 1
    w = np.array([[w1], [w2]])
    workload_cov = np.eye(2)
    neg_log_discount_factor = np.log(0.95)
    si_object = StrategicIdlingHedging(
        workload_mat=workload_mat,
        workload_cov=workload_cov,
        neg_log_discount_factor=neg_log_discount_factor,
        load=load,
        cost_per_buffer=cost_per_buffer,
        model_type=env.model_type,
        strategic_idling_params=StrategicIdlingParams())
    hedging_decision_dict = si_object._non_negative_workloads(w)
    hedging_tuple = si_object._get_null_strategic_idling_output(**hedging_decision_dict)
    assert hedging_tuple.beta_star == 0
    assert hedging_tuple.k_idling_set.size == 0
    assert hedging_tuple.sigma_2_h == 0
    assert hedging_tuple.psi_plus is None
    assert np.all(hedging_tuple.w_star == w)
    assert hedging_tuple.c_plus is None
    assert len(si_object.psi_plus_cone_list) == 0
    assert len(si_object.beta_star_cone_list) == 0


def test_hedging_inside_monotone_region_when_it_is_a_ray():
    """We use the simple re-entrant line with parameters that make monotone region to be a ray."""
    state = np.array([[302], [297], [300]])
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.array([[1], [2], [3]]))
    num_wl_vec = 2
    neg_log_discount_factor = - np.log(0.95)
    workload_cov = 10 * np.eye(num_wl_vec)
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    si_object = StrategicIdlingHedging(
        workload_mat=workload_mat,
        workload_cov=workload_cov,
        neg_log_discount_factor=neg_log_discount_factor,
        load=load,
        cost_per_buffer=env.cost_per_buffer,
        model_type=env.model_type,
        strategic_idling_params=StrategicIdlingParams())

    w = workload_mat @ state  # = np.array([[59.9], [54.59090909]])
    w_star_ray = si_object._find_workload_with_min_eff_cost_by_idling(w)
    # We start right in the ray

    hedging_decision_dict = si_object._non_negative_workloads(w_star_ray)
    hedging_tuple = si_object._get_null_strategic_idling_output(**hedging_decision_dict)

    assert hedging_tuple.beta_star == 0
    assert hedging_tuple.k_idling_set.size == 0
    assert hedging_tuple.sigma_2_h == 0
    assert hedging_tuple.psi_plus is None
    assert np.allclose(hedging_tuple.w_star, w_star_ray)
    assert hedging_tuple.c_plus is None
    assert len(si_object.psi_plus_cone_list) == 0
    assert len(si_object.beta_star_cone_list) == 0


def test_hedging_inside_monotone_region_actually_in_its_boundary():
    """We use the simple reentrant line."""
    state = np.array([[1], [1], [1]])
    env = examples.simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                               cost_per_buffer=np.ones((3, 1)))
    num_wl_vec = 2
    neg_log_discount_factor = - np.log(0.95)
    workload_cov = 10 * np.eye(num_wl_vec)
    load, workload_mat, nu = wl.compute_load_workload_matrix(env, num_wl_vec)
    w = np.dot(workload_mat, state)

    si_object = StrategicIdlingHedging(
        workload_mat=workload_mat,
        workload_cov=workload_cov,
        neg_log_discount_factor=neg_log_discount_factor,
        load=load,
        cost_per_buffer=env.cost_per_buffer,
        model_type=env.model_type,
        strategic_idling_params=StrategicIdlingParams())

    hedging_decision_dict = si_object._non_negative_workloads(w)
    hedging_tuple = si_object._get_null_strategic_idling_output(**hedging_decision_dict)

    assert hedging_tuple.beta_star == 0
    assert hedging_tuple.k_idling_set.size == 0
    assert hedging_tuple.sigma_2_h == 0
    assert hedging_tuple.psi_plus is None
    assert hedging_tuple.w_star is not None
    assert hedging_tuple.c_plus is None
    assert len(si_object.psi_plus_cone_list) == 0
    assert len(si_object.beta_star_cone_list) == 0


def get_idling_directions_with_w_as_input(w, si_object):
    idling_decision_dict = si_object._non_negative_workloads(w)
    idling_decision_dict = si_object._add_standard_hedging(w, idling_decision_dict)
    hedging_tuple = si_object._get_null_strategic_idling_output(**idling_decision_dict)
    return hedging_tuple


def test_hedging_thresdhold_value_when_monotone_region_is_a_ray():
    """Set multple w such that we are always below the same face, and such that a path-wise optimal
    solution starting from w cannot exist. This is achieved e.g. for the simple reentrant line when
    w=(w1, w2), and w2 = load[1] / load[0] * w1 * 0.9 (p. 187, CTCN online ed)."""
    epsilon = 0.1
    discount_factor = 0.95
    workload_cov = np.eye(2)
    num_tests = 100
    strategic_idling_params = StrategicIdlingParams(epsilon=epsilon)

    # Environment parameters
    alpha1 = 0.33
    mu_1 = 0.66 * 1.03
    mu_2 = 0.33 * 1.05
    mu_3 = mu_1
    c_1 = 1
    c_2 = 2
    c_3 = 3
    cost_per_buffer = np.array([[c_1], [c_2], [c_3]])
    env = examples.simple_reentrant_line_model(alpha1=alpha1, mu1=mu_1, mu2=mu_2, mu3=mu_3,
                                               cost_per_buffer=cost_per_buffer)

    # We use the following in order to compute the numerical values:
    # epsilon = 0.1
    workload_mat = np.array([[1 / mu_1 + 1 / mu_3, 1 / mu_3, 1 / mu_3],
                             [1 / mu_2, 1 / mu_2, 0]])
    c_plus_theory = np.array([[mu_1 * (c_1 - c_2)],
                              [mu_2 * (c_2 * (1 + mu_1/mu_3) - c_1 * mu_1/mu_3)]])
    c_minus_theory = np.array([[mu_3 * c_3],
                               [mu_2 * (c_1 - c_3 * (1 + mu_3 / mu_1))]])
    psi_plus = c_plus_theory - c_minus_theory
    # Since it is a ray, we replace c_plus with a multiple of w_star
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)
    w = np.array([1, load[1] / load[0] * 0.9])[:, None]
    w_star = np.array([w[0], - w[0] * psi_plus[0] / psi_plus[1]])
    epsilon_naught = (c_minus_theory.T @ w_star) / (np.linalg.norm(w_star) ** 2)
    c_plus_new = (epsilon_naught + epsilon / np.linalg.norm(w_star)) * w_star
    psi_plus_new = c_plus_new - c_minus_theory

    # Using the equations from Cookbook as included in this spreadsheet
    # https://docs.google.com/spreadsheets/d/1Vaaq5LX4zlrEo1Sk14MX0eQfVaBLC3XsJtSVlG29JRA:
    lambda_star_theory = 0.1216490057
    delta_h_theory = 0.04164551275
    sigma_2_h_theory = 7.111831823
    theta_roots_theory = 0.1261016697
    beta_star_theory = 1.028606519

    neg_log_discount_factor = - np.log(discount_factor)

    si_object = create_strategic_idling_object(workload_mat=workload_mat, workload_cov=workload_cov,
                                               neg_log_discount_factor=neg_log_discount_factor,
                                               load=load,
                                               cost_per_buffer=env.cost_per_buffer,
                                               strategic_idling_params=strategic_idling_params)

    beta_star_simul, sigma_2_h_simul, delta_h_simul, lambda_star_simul, theta_roots_simul = \
        si_object._compute_hedging_threshold(c_plus_new, psi_plus_new)
    np.testing.assert_almost_equal(beta_star_simul, beta_star_theory, decimal=5)
    np.testing.assert_almost_equal(sigma_2_h_simul, sigma_2_h_theory, decimal=5)
    np.testing.assert_almost_equal(delta_h_simul, delta_h_theory, decimal=5)
    np.testing.assert_almost_equal(lambda_star_simul, lambda_star_theory, decimal=5)
    np.testing.assert_almost_equal(theta_roots_simul, theta_roots_theory, decimal=5)

    for i in range(num_tests):
        w1 = i + 1
        w2 = load[1] / load[0] * w1 * 0.9
        w = np.array([[w1], [w2]])

        si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                                   workload_cov=workload_cov,
                                                   neg_log_discount_factor=neg_log_discount_factor,
                                                   load=load,
                                                   cost_per_buffer=env.cost_per_buffer,
                                                   strategic_idling_params=strategic_idling_params)

        hedging_tuple = get_idling_directions_with_w_as_input(w, si_object)

        np.testing.assert_almost_equal(hedging_tuple.beta_star, beta_star_simul, decimal=4)
        assert np.all(hedging_tuple.k_idling_set == np.array([1]))
        np.testing.assert_almost_equal(hedging_tuple.c_plus, c_plus_new)
        np.testing.assert_almost_equal(hedging_tuple.psi_plus, psi_plus_new, decimal=4)
        assert len(si_object.psi_plus_cone_list) == 0  # This is not a demand model
        assert len(si_object.beta_star_cone_list) == 0  # This is not a demand model


def test_hedging_when_no_monotone_region_but_only_boundary_of_feasible_workload_space():
    """We use the single server queue with demand model, when there is only one c_bar vector."""
    discount_factor = 0.95

    c_s = 1
    c_p = 5
    c_d = 10
    cost_per_buffer = np.array([[c_s], [c_p], [c_d]])
    mu_s = 50
    mu_p = 10
    mu_d = 11
    env = examples.single_station_demand_model(alpha_d=9, mu=mu_p, mus=mu_s, mud=mu_d,
                                               cost_per_buffer=cost_per_buffer)
    workload_cov = np.eye(2)
    load, workload_mat, nu = wl.compute_load_workload_matrix(env)

    w = np.array([[-0.09], [0.1]])

    epsilon = 0.01  # Defines the width of the artificial cone around the infeasible region.
    strategic_idling_params = StrategicIdlingParams(epsilon=epsilon)

    # We replace c_plus with a multiple of w_star, and use this to obtain psi_plus.
    # Using the equations from Cookbook as included in this spreadsheet
    # https://docs.google.com/spreadsheets/d/1Vaaq5LX4zlrEo1Sk14MX0eQfVaBLC3XsJtSVlG29JRA:
    # w_star_theory = np.array([[0.11], [0.1]]) = (w[1]*mu_d/mu_p, w[1]), from workload_mat columns
    # c_minus = np.array([[-0.09090909091], [3]])  # c_bar = (-mu_p * c_p, mu_d * (c_d + c_p))
    psi_plus_theory = np.array([[1.911965423], [-2.088295896]])
    # lambda_star_theory = 0.5245254448
    # delta_h_theory = -0.1884936207
    # sigma_2_h_theory = 8.01659153
    # theta_theory = 0.09202771306
    beta_star_theory = 8.078456862
    neg_log_discount_factor = - np.log(discount_factor)
    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               workload_cov=workload_cov,
                                               neg_log_discount_factor=neg_log_discount_factor,
                                               load=load,
                                               cost_per_buffer=env.cost_per_buffer,
                                               model_type=env.model_type,
                                               strategic_idling_params=strategic_idling_params)

    hedging_tuple = get_idling_directions_with_w_as_input(w, si_object)

    np.testing.assert_almost_equal(hedging_tuple.beta_star, beta_star_theory, decimal=3)
    np.testing.assert_almost_equal(hedging_tuple.psi_plus, psi_plus_theory)
    assert len(si_object.psi_plus_cone_list) == 1  # Added to the cone envelope
    assert len(si_object.beta_star_cone_list) == 1  # Added to the cone envelope
    np.testing.assert_almost_equal(si_object.psi_plus_cone_list[0], psi_plus_theory, decimal=3)
    np.testing.assert_almost_equal(si_object.beta_star_cone_list[0], [beta_star_theory],
                                   decimal=3)


def test_standard_hedging_negative_orthant():
    w = np.zeros((3, 1))
    w[0] = -1
    workload_mat = np.eye(3)
    workload_cov = np.eye(3)
    neg_log_discount_factor = 0.05
    load = 0.95 * np.ones((3, 1))
    cost_per_buffer = np.ones((3, 1))
    psi_plus = np.ones((3, 1))
    beta_star = 1
    with pytest.raises(AssertionError):
        si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                                   workload_cov=workload_cov,
                                                   neg_log_discount_factor=neg_log_discount_factor,
                                                   load=load,
                                                   cost_per_buffer=cost_per_buffer)
        si_object._psi_plus_cone_list = [psi_plus]
        si_object._beta_star_cone_list = [beta_star]
        _ = si_object._non_negative_workloads(w)


def test_negative_orthant_hedging_with_nonnegative_w():
    w = np.zeros((3, 1))
    w[0] = 1
    psi_plus = np.ones((3, 1))
    beta_star = 1
    workload_mat = np.zeros((3, 2))
    cost_per_buffer = np.zeros((2, 1))

    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    si_object._psi_plus_cone_list = [psi_plus]
    si_object._beta_star_cone_list = [beta_star]

    with pytest.raises(AssertionError):
        _ = si_object._negative_workloads(w)


def test_negative_orthant_empty_lists():
    psi_plus_cone_list = []
    beta_star_cone_list = []
    w = np.array([[-1], [-0.5]])
    workload_mat = np.zeros((3, 2))
    cost_per_buffer = np.zeros((2, 1))

    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)
    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list

    with pytest.raises(AssertionError):
        _ = si_object._negative_workloads(w)


def test_negative_orthant_workloads_with_some_zero_components():
    psi_plus_cone_list = [np.array([[1], [-0.5]]), np.array([[1], [0.5]])]
    beta_star_cone_list = [0, 0]
    workload_mat = np.array([[0, -1, 1], [0, 0, 1]])
    cost_per_buffer = np.ones((3, 1))

    state = np.array([[0], [1], [0]])
    w = workload_mat @ state  # w = np.array([[-1], [0]])

    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)

    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_decision_dict = si_object._negative_workloads(w)
    si_tuple = si_object._get_null_strategic_idling_output(**si_decision_dict)

    assert np.all(si_tuple.k_idling_set == [0, 1])


def test_negative_orthant_zero_beta_star():
    psi_plus_cone_list = [np.array([[1], [-0.5]]), np.array([[1], [0.5]])]
    beta_star_cone_list = [0, 0]
    workload_mat = np.array([[0, -1, 1], [0, 0, 1]])
    cost_per_buffer = np.ones((3, 1))

    state = np.array([[0], [2], [0]])
    w_1 = workload_mat @ state

    si_object = create_strategic_idling_object(workload_mat=workload_mat,
                                               cost_per_buffer=cost_per_buffer)

    si_object._psi_plus_cone_list = psi_plus_cone_list
    si_object._beta_star_cone_list = beta_star_cone_list
    si_decision_dict = si_object._negative_workloads(w_1)
    si_tuple = si_object._get_null_strategic_idling_output(**si_decision_dict)

    assert np.all(si_tuple.k_idling_set == [0, 1])


def test_negative_orthant_positive_beta_star():
    psi_plus_cone_list = [np.array([[1], [-0.5]])]
    beta_star_cone_list = [5]
    workload_mat = np.array([[0, -1, 1], [0, 0, 1]])
    cost_per_buffer = np.ones((3, 1))

    state = np.array([[0], [1], [0]])
    w_1 = workload_mat @ state

    si_object_1 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer)

    si_object_1._psi_plus_cone_list = psi_plus_cone_list
    si_object_1._beta_star_cone_list = beta_star_cone_list
    si_decision_dict_1 = si_object_1._negative_workloads(w_1)
    si_tuple_1 = si_object_1._get_null_strategic_idling_output(**si_decision_dict_1)

    assert si_tuple_1.k_idling_set.size == 0

    state = np.array([[0], [10], [0]])
    w_2 = workload_mat @ state

    si_object_2 = create_strategic_idling_object(workload_mat=workload_mat,
                                                 cost_per_buffer=cost_per_buffer)

    si_object_2._psi_plus_cone_list = psi_plus_cone_list
    si_object_2._beta_star_cone_list = beta_star_cone_list
    si_decision_dict_2 = si_object_2._negative_workloads(w_2)
    si_tuple_2 = si_object_2._get_null_strategic_idling_output(**si_decision_dict_2)

    assert np.all(si_tuple_2.k_idling_set == [0, 1])
