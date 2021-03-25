""" Test of the function compute_dual_effective_cost from hedging.py for various models.

    We compare the result of the function with theoretical results based on computation by hand. To
    do the computation by hand, you have to compute the workload matrix (see docstring in the class
    TestComputeNetworkLoadAndWorkload in test_workload.py for more information). Then, you have to
    write the system of inequalities defined by workload_mat.T barc <= cost_per_buffer. This system
    defines a feasible region for barc and you have to compute the vertexes of that region, i.e the
    intersection between the halfspaces defined by each inequality. Then, you have to find for which
    values of mu and cost_per_buffer each vertex is feasible (i.e does it satisfies the other
    inequalities). Finally, given the list of feasible vertexes you have to find which one realise
    the maximum in \max_{i} \{w^T \bar{c}^i\} for a specific w. """

import numpy as np
import src.snc.environments.examples as examples
from src import snc as wl
from src.snc.utils import alt_methods_test

import pytest


class TestSingleStationDemandModel:
    """ Test for the single station demand model for specific cases. """

    @staticmethod
    def perform_test(alpha_d, mu, mus, mud, cost_per_buffer, num_wl_vec, w):
        """ Test for the single station demand model (see example 7.1.2 from CTCN book)
        given parameters of the environment, the relaxation and a specific point w. """
        assert num_wl_vec in [1, 2, 3]

        env = examples.single_station_demand_model(alpha_d=alpha_d, mu=mu, mus=mus, mud=mud,
                                                   cost_per_buffer=cost_per_buffer)

        sim_time_interval = env.job_generator.sim_time_interval
        # We define the theoretical workload matrix and compare it to the one we compute in order to
        # find which relaxation we are doing
        # Theoretical workload matrix and load
        workload_mat_theory = np.array([[-1. / mus, -1. / mus, 1. / mus],
                                        [0., -1. / mu, 1. / mu],
                                        [0., 0., 1. / mud]])
        workload_mat_theory = np.multiply(workload_mat_theory, 1./sim_time_interval)
        load_theory = np.array([alpha_d / mus, alpha_d / mu, alpha_d / mud])
        # Computed workload matrix (sorted by load)
        _, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)

        # Theoretical vertexes of the \bar{c} feasible region based on the dim of the relaxation
        vertexes_1d = np.array([[-mus * cost_per_buffer[0]],
                                [mus * cost_per_buffer[2]],
                                [-mus * cost_per_buffer[1]],
                                [-mu * cost_per_buffer[1]],
                                [mu * cost_per_buffer[2]],
                                [mud * cost_per_buffer[2]]])
        vertexes_1d = np.multiply(vertexes_1d, sim_time_interval)
        vertexes_2d = np.array([[-mus * cost_per_buffer[0],
                                 -mu * (cost_per_buffer[1] - cost_per_buffer[0])],
                                [-mus * cost_per_buffer[0],
                                 mu * (cost_per_buffer[2] + cost_per_buffer[0])]])
        vertexes_2d = np.multiply(vertexes_2d, sim_time_interval)
        vertexes_3d = np.array([[-mus * cost_per_buffer[0],
                                 -mu * (cost_per_buffer[1] - cost_per_buffer[0]),
                                 mud * (cost_per_buffer[2] + cost_per_buffer[1])]])
        vertexes_3d = np.multiply(vertexes_3d, sim_time_interval)
        # We select which vertexes are feasible based on the relaxation and the env parameters
        if num_wl_vec == 1:
            if np.allclose(workload_mat, workload_mat_theory[[0], :]):
                if -mus * cost_per_buffer[0] >= -mus * cost_per_buffer[1]:
                    feasible_vertexes = vertexes_1d[[0, 1], :, :]
                else:
                    feasible_vertexes = vertexes_1d[[2, 1], :, :]
            elif np.allclose(workload_mat, workload_mat_theory[[1], :]):
                feasible_vertexes = vertexes_1d[[3, 4], :, :]
            elif np.allclose(workload_mat, workload_mat_theory[[2], :]):
                feasible_vertexes = vertexes_1d[[5], :, :]
            else:
                raise ValueError("The workload_mat returned by compute_load_workload_matrix is \
                                 wrong")
        elif num_wl_vec == 2:
            # The theoretical \bar{c} vectors were computed for a specific order of the workload
            # vectors. So we compute sort_by_load_index to be able to reorder the theoretical
            # \bar{c} components based on the sort made by load
            sort_by_load_index = np.argsort(load_theory[[0, 1]])[::-1]
            # The test is only implemented when we relax the demand workload vector
            np.testing.assert_almost_equal(workload_mat,
                                           workload_mat_theory[[0, 1], :][sort_by_load_index, :])
            feasible_vertexes = vertexes_2d[[0, 1], :, :]
            feasible_vertexes = feasible_vertexes[:, sort_by_load_index, :]
        else:
            feasible_vertexes = vertexes_3d[[0], :, :]
            # The theoretical \bar{c} vectors were computed for a specific order of the workload
            # vectors. So we compute sort_by_load_index to be able to reorder the theoretical
            # \bar{c} components based on the sort made by load
            sort_by_load_index = np.argsort(load_theory)[::-1]
            feasible_vertexes = feasible_vertexes[:, sort_by_load_index, :]
        # Compute the index of the theoretical vertex which satisfy the max
        max_vertex_index = np.argmax(np.dot(w.T, feasible_vertexes).flatten())
        barc_theory = feasible_vertexes[max_vertex_index]
        barc, _, _ = alt_methods_test.compute_dual_effective_cost_cvxpy(w, workload_mat,
                                                                        cost_per_buffer, method='cvx.ECOS')

        np.testing.assert_almost_equal(barc, barc_theory, decimal=4)

    def test_single_station_demand_model_case1(self):
        # 1D relaxation, we only keep supply workload vector,
        # -mus * cost_per_buffer[0] >= -mus * cost_per_buffer[1]
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e2,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[-1.]]))

    def test_single_station_demand_model_case2(self):
        # 1D relaxation, we only keep supply workload vector,
        # -mus * cost_per_buffer[0] >= -mus * cost_per_buffer[1]
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e2,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[1.]]))

    def test_single_station_demand_model_case3(self):
        # 1D relaxation, we only keep supply workload vector,
        # -mus * cost_per_buffer[0] < -mus * cost_per_buffer[1]
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e2,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[-1.]]))

    def test_single_station_demand_model_case4(self):
        # 1D relaxation, we only keep supply workload vector,
        # -mus * cost_per_buffer[0] < -mus * cost_per_buffer[1]
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e2,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[1.]]))

    def test_single_station_demand_model_case5(self):
        # 1D relaxation, we only keep processing workload vector
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=10, mus=1e2, mud=1e2,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[-1.]]))

    def test_single_station_demand_model_case6(self):
        # 1D relaxation, we only keep processing workload vector
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=10, mus=1e2, mud=1e2,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[1.]]))

    def test_single_station_demand_model_case7(self):
        # 1D relaxation, we only keep demand workload vector
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=1e2, mud=10,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=1, w=np.array([[1.]]))

    @pytest.mark.skip("""For this w, the test fails because this w is exactly on the intersection 
    between the cone spanned by barc_1_theory and the w infeasible region. Thus, this w is 
    orthogonal to the face of the feasible region of \bar{c} which is defined by the vertex 
    barc_1_theory and the intersection between the feasible region of \bar{c} and the infeasible 
    region of w. So, the simplex method can give any of the two vertexes but they should have the 
    same effective cost.""")
    def test_single_station_demand_model_case8(self):
        # 2D relaxation, we relax demand workload vector, the first theoretical \bar{c} of the
        # feasible vertexes has a negative second component
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=2, w=np.array([[-10.],
                                                                            [-1.]]))

    def test_single_station_demand_model_case9(self):
        # 2D relaxation, we relax demand workload vector, the first theoretical \bar{c} of the
        # feasible vertexes has a negative second component
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=2,
                                                  w=np.array([[-10. - 1e-2],
                                                              [-1.]]))

    def test_single_station_demand_model_case10(self):
        # 2D relaxation, we relax demand workload vector, the first theoretical \bar{c} of the
        # feasible vertexes has a negative second component
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_single_station_demand_model_case11(self):
        # 2D relaxation, we relax demand workload vector, the first theoretical \bar{c} of the
        # feasible vertexes has a positive second component
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [3.]]),
                                                  num_wl_vec=2, w=np.array([[-30],
                                                                            [-1.]]))

    def test_single_station_demand_model_case12(self):
        # 2D relaxation, we relax demand workload vector, the first theoretical \bar{c} of the
        # feasible vertexes has a positive second component
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [3.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_single_station_demand_model_case13(self):
        # No relaxation, the theoretical \bar{c} has a negative second component
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [3.]]),
                                                  num_wl_vec=3, w=np.array([[1.],
                                                                            [1.],
                                                                            [1.]]))

    def test_single_station_demand_model_case14(self):
        # No relaxation, the theoretical \bar{c} has a positive second component
        TestSingleStationDemandModel.perform_test(alpha_d=9, mu=1e2, mus=10, mud=1e3,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [3.]]),
                                                  num_wl_vec=3, w=np.array([[1.],
                                                                            [1.],
                                                                            [1.]]))


class TestSimpleReentrantLineModel:
    """ Test for the simple re-entrant line for specific cases. """

    @staticmethod
    def perform_test(alpha1, mu1, mu2, mu3, cost_per_buffer, num_wl_vec, w):
        """ Test for the simple re-entrant line (see example 4.2.3, Figure 2.9  from CTCN book)
        given parameters of the environment, the relaxation and a specific point w. """

        assert num_wl_vec == 2  # The test is only implemented without relaxation

        env = examples.simple_reentrant_line_model(alpha1=alpha1, mu1=mu1, mu2=mu2, mu3=mu3,
                                                   cost_per_buffer=cost_per_buffer)

        # We define the theoretical workload matrix and compare it to the one we compute in order to
        # find which relaxation we are doing
        # Theoretical workload matrix and load
        workload_mat_theory = np.array([[1. / mu1 + 1. / mu3, 1. / mu3, 1. / mu3],
                                        [1. / mu2, 1. / mu2, 0.]])
        load_theory = np.array([alpha1 / mu1 + alpha1 / mu3, alpha1 / mu2])
        # Computed workload matrix (sorted by load)
        _, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)

        # Theoretical vertexes of the \bar{c} feasible region based on the dim of the relaxation
        vertexes_2d = np.array([[mu3 * cost_per_buffer[2],
                                 mu2 * cost_per_buffer[0] - cost_per_buffer[2] * mu2
                                 * (mu3 / mu1 + 1.)],
                                [mu1 * (cost_per_buffer[0] - cost_per_buffer[1]),
                                 mu2 * cost_per_buffer[1] + (mu1 * mu2 / mu3) *
                                 (cost_per_buffer[1] - cost_per_buffer[0])],
                                [mu3 * cost_per_buffer[2],
                                 mu2 * (cost_per_buffer[1] - cost_per_buffer[2])]])
        # We select which vertexes are feasible based on the env parameters
        if mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2]:
            feasible_vertexes = vertexes_2d[[0, 1], :, :]
        else:
            feasible_vertexes = vertexes_2d[[2], :, :]
        # The theoretical \bar{c} vectors were computed for a specific order of the workload
        # vectors. So we compute sort_by_load_index to be able to reorder the theoretical
        # \bar{c} components based on the sort made by load
        sort_by_load_index = np.argsort(load_theory)[::-1]
        feasible_vertexes = feasible_vertexes[:, sort_by_load_index, :]
        # Compute the index of the theoretical vertex which satisfy the max
        max_vertex_index = np.argmax(np.dot(w.T, feasible_vertexes).flatten())
        barc_theory = feasible_vertexes[max_vertex_index]
        barc, _, _ = alt_methods_test.compute_dual_effective_cost_cvxpy(w, workload_mat,
                                                                        cost_per_buffer, method='cvx.ECOS')
        np.testing.assert_almost_equal(barc, barc_theory, decimal=4)

    def test_simple_reentrant_line_model_case1(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) > mu3 * cost_per_buffer[2],
        # the second component of the theoretical \bar{c} is positive
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[4.],
                                                                            [2.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case2(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) > mu3 * cost_per_buffer[2],
        # the second component of the theoretical \bar{c} is negative
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[4.],
                                                                            [1.],
                                                                            [2.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case3(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has a positive
        # first component and a negative second component, the second theoretical \bar{c} of the
        # feasible vertexes has all positive components
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[2.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case4(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has a positive
        # first component and a negative second component, the second theoretical \bar{c} of the
        # feasible vertexes has all positive components
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[2.],
                                                                            [1.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case5(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has a positive
        # first component and a negative second component, the second theoretical \bar{c} of the
        # feasible vertexes has a negative first component and a positive second component
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[2.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case6(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has a positive
        # first component and a negative second component, the second theoretical \bar{c} of the
        # feasible vertexes has a negative first component and a positive second component
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[1.],
                                                                            [2.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case7(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has all
        # positive components, the second theoretical \bar{c} of the feasible vertexes has a
        # negative first component and a positive second component
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[3.],
                                                                            [4.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[2.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case8(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has all
        # positive components, the second theoretical \bar{c} of the feasible vertexes has a
        # negative first component and a positive second component
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[3.],
                                                                            [4.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case9(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has all positive
        # components, the second theoretical \bar{c} of the feasible vertexes has all positive
        # components
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[3.],
                                                                            [2.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[2.],
                                                                            [1.]]))

    def test_simple_reentrant_line_model_case10(self):
        # mu1 * (cost_per_buffer[0] - cost_per_buffer[1]) <= mu3 * cost_per_buffer[2],
        # the first theoretical \bar{c} of the feasible vertexes has all positive
        # components, the second theoretical \bar{c} of the feasible vertexes has all positive
        # components
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineModel.perform_test(alpha1=0.3, mu1=0.68, mu2=0.35, mu3=0.68,
                                                  cost_per_buffer=np.array([[3.],
                                                                            [2.],
                                                                            [1.]]),
                                                  num_wl_vec=2, w=np.array([[1.],
                                                                            [1.]]))


class TestSimpleReentrantLineWithDemandModel:
    """ Test for the simple re-entrant line with demand for specific cases. """

    @staticmethod
    def perform_test(alpha_d, mu1, mu2, mu3, mus, mud, cost_per_buffer, num_wl_vec, w):
        """ Test for the simple re-entrant line with demand (see example 7.5.1. Simple re-entrant
        line with demand, Figure 7.5  from CTCN book) given parameters of the environment, the
        relaxation and a specific point w. """

        assert num_wl_vec == 2  # The test is only implemented for 2D relaxation

        env = examples.simple_reentrant_line_with_demand_model(alpha_d=alpha_d, mu1=mu1, mu2=mu2,
                                                               mu3=mu3, mus=mus, mud=mud,
                                                               cost_per_buffer=cost_per_buffer)

        # We define the theoretical workload matrix and compare it to the one we compute in order to
        # find which relaxation we are doing
        # Theoretical workload matrix and load
        workload_mat_theory = np.array([[-1. / mus, -1. / mus, -1. / mus, -1. / mus, 1. / mus],
                                        [0., -1. / mu1, -1. / mu1, -1. / mu1 - 1. / mu3,
                                         1. / mu1 + 1. / mu3],
                                        [0., 0., -1. / mu2, -1. / mu2, 1. / mu2],
                                        [0., 0., 0., 0., 1. / mud]])
        load_theory = np.array([alpha_d / mus,
                                alpha_d / mu1 + alpha_d / mu3,
                                alpha_d / mu2,
                                alpha_d / mud])
        # Computed workload matrix (sorted by load)
        _, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)

        # Theoretical vertexes of the \bar{c} feasible region based on the dim of the relaxation
        vertexes_2d = np.array([[mu3 * (cost_per_buffer[4] + cost_per_buffer[2]),
                                 -(mu2 * mu3 / mu1) * (cost_per_buffer[4] + (1. + mu1 / mu3)
                                                       * cost_per_buffer[2])],
                                [-mu1 * cost_per_buffer[1],
                                 mu2 * (cost_per_buffer[4] + (1. + mu1 / mu3) * cost_per_buffer[
                                     1])],
                                [mu3 * (cost_per_buffer[2] - cost_per_buffer[3]),
                                 (mu2 * mu3 / mu1) * (cost_per_buffer[3] - (1. + mu1 / mu3)
                                                      * cost_per_buffer[2])],
                                [-mu1 * cost_per_buffer[1],
                                 -mu2 * (cost_per_buffer[3] - (1. + mu1 / mu3)
                                         * cost_per_buffer[1])],
                                [-mu1 * cost_per_buffer[1],
                                 -mu2 * (cost_per_buffer[2] - cost_per_buffer[1])]])
        # We select which vertexes are feasible based on the env parameters
        if mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) >= -mu1 * cost_per_buffer[1]:
            feasible_vertexes = vertexes_2d[[0, 1, 2, 3], :, :]
        else:
            feasible_vertexes = vertexes_2d[[0, 1, 4], :, :]
        # The theoretical \bar{c} vectors were computed for a specific order of the workload
        # vectors. So we compute sort_by_load_index to be able to reorder the theoretical
        # \bar{c} components based on the sort made by load
        sort_by_load_index = np.argsort(load_theory[[1, 2]])[::-1]
        # The test is only implemented when we relax the demand  and the supply workload vectors
        np.testing.assert_almost_equal(workload_mat,
                                       workload_mat_theory[[1, 2], :][sort_by_load_index, :])
        feasible_vertexes = feasible_vertexes[:, sort_by_load_index, :]
        # Compute the index of the theoretical vertex which satisfy the max
        max_vertex_index = np.argmax(np.dot(w.T, feasible_vertexes).flatten())
        barc_theory = feasible_vertexes[max_vertex_index]
        barc, _, _ = alt_methods_test.compute_dual_effective_cost_cvxpy(w, workload_mat,
                                                                        cost_per_buffer, method='cvx.ECOS')
        np.testing.assert_almost_equal(barc, barc_theory, decimal=4)

    def test_simple_reentrant_line_with_demand_model_case1(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is negative
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2,
                                                            w=np.array([[2.], [1.]]))

    def test_simple_reentrant_line_with_demand_model_case2(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is negative
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2,
                                                            w=np.array([[1.], [1.]]))

    def test_simple_reentrant_line_with_demand_model_case3(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is negative
        # For this w, the theoretical \bar{c} should be the third one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2,
                                                            w=np.array([[-2.], [-1.]]))

    def test_simple_reentrant_line_with_demand_model_case4(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [2.],
                                                                                      [1.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[2.], [1.]]))

    def test_simple_reentrant_line_with_demand_model_case5(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [2.],
                                                                                      [1.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[1.], [1.]]))

    def test_simple_reentrant_line_with_demand_model_case6(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) < -mu1 * cost_per_buffer[1],
        # the second component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the fourth one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [2.],
                                                                                      [1.], [3.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[-2.],
                                                                                      [-1.]]))

    def test_simple_reentrant_line_with_demand_model_case7(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) >= -mu1 * cost_per_buffer[1],
        # the first component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the first one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [1.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[2.], [1.]]))

    def test_simple_reentrant_line_with_demand_model_case8(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) >= -mu1 * cost_per_buffer[1],
        # the first component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the second one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [1.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[1.], [2.]]))

    def test_simple_reentrant_line_with_demand_model_case9(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) >= -mu1 * cost_per_buffer[1],
        # the first component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the third one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [1.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[-6.],
                                                                                      [-10.]]))

    def test_simple_reentrant_line_with_demand_model_case10(self):
        # mu3 * (cost_per_buffer[2] - cost_per_buffer[3]) >= -mu1 * cost_per_buffer[1],
        # the first component of the third theoretical \bar{c} of the feasible
        # vertexes is positive
        # For this w, the theoretical \bar{c} should be the fourth one of the feasible vertexes
        TestSimpleReentrantLineWithDemandModel.perform_test(alpha_d=.09, mu1=.19, mu2=.1, mu3=.2,
                                                            mus=1, mud=1,
                                                            cost_per_buffer=np.array([[1.], [1.],
                                                                                      [2.], [1.],
                                                                                      [1.]]),
                                                            num_wl_vec=2, w=np.array([[-2.],
                                                                                      [-1.]]))
