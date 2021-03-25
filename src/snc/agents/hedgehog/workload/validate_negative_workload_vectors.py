from typing import Tuple

import cvxpy as cvx
import numpy as np
from snc.utils.snc_tools import is_binary
import snc.utils.snc_types as types


class ValidateNegativeWorkloadVectors:
    def __init__(self,
                 constituency_matrix: types.ConstituencyMatrix,
                 buffer_processing_matrix: types.BufferMatrix,
                 workload_mat: types.WorkloadMatrix,
                 tol: float = 1e-4,
                 convex_solver: str = 'cvx.CPLEX') -> None:
        """
        When obtaining the workload vectors (either from the vertexes of the feasible region of the
        network load dual problem or by transforming the definition of the velocity space into a
        convex hull formulation, (6.18) and (6.11) in CTCN book, respectively), we can obtain
        vectors with negative components for both push and pull models. This class includes methods
        to distinguish whether these vectors with negative components might refer to draining
        velocities, hence they are useful; or only to filling velocities, hence they can be
        discharged.

        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :param workload_mat: Workload matrix, with rows being workload vectors.
        :param tol: Tolerance to detect negative components.
        :param convex_solver: String indicating which convex solver should be used.
        :return: None.
        """
        assert constituency_matrix.shape[1] == buffer_processing_matrix.shape[1]
        assert buffer_processing_matrix.shape[0] == workload_mat.shape[1]

        assert tol >= 0
        self.tol = tol
        self.convex_solver = convex_solver
        self._lp_problem, self._xi, self._z, self._h = self.create_lp_find_vertex_velocity_space(
            constituency_matrix, buffer_processing_matrix)

    @staticmethod
    def create_lp_find_vertex_velocity_space(
            constituency_matrix: types.ConstituencyMatrix,
            buffer_processing_matrix: types.BufferMatrix
    ) -> Tuple[cvx.Problem, cvx.Parameter, cvx.Variable, cvx.Variable]:
        """
        Create a parameterised LP that computes the minimum value greater than all components of the
        velocity space corresponding with a parameterised workload vector. The LP will be solved for
        each workload vectors with negative components.

        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :return: (lp_problem, xi, z, v)
            - lp_problem: LP program as a CVX problem.
            - xi: Workload vector as a parameter to be passed before solving the problem.
            - z: Action variable to achieve the velocity that has all its components smaller than h.
            - h: Minimum value of all components of the velocity space given by xi.

        """
        num_buffers, num_activities = buffer_processing_matrix.shape
        z = cvx.Variable((num_activities, 1), nonneg=True)
        h = cvx.Variable()
        xi = cvx.Parameter((num_buffers, 1))
        objective = cvx.Minimize(h)
        constraints = [
            xi.T @ (buffer_processing_matrix @ z) == -1,
            buffer_processing_matrix @ z <= h * np.ones((num_buffers, 1)),
            constituency_matrix @ z <= 1
        ]
        lp_problem = cvx.Problem(objective, constraints)
        return lp_problem, xi, z, h

    def is_draining_workload_vector(self, xi: types.WorkloadSpace) -> int:
        """
        Check if the current workload vector is draining, hence valid; or filling, so useless.

        :param xi: Workload vector to be evaluated.
        :return Yes/No boolean value.
        """
        self._xi.value = xi
        self._lp_problem.solve(solver=eval(self.convex_solver), warm_start=True)
        return 1 if self._h.value <= - self.tol else 0

    def identify_draining_workload_vectors(self, workload_mat: types.WorkloadMatrix) \
            -> types.WorkloadSpace:
        """
        Returns a binary array indexing which workload vectors (i.e., rows of the workload matrix)
        are draining ones and therefore valid.

        :param workload_mat: Workload matrix, with rows being workload vectors.
        :return draining_workload_vec: Binary array with ones indexing valid workload vectors.
        """
        num_wl_vec = workload_mat.shape[0]
        draining_workload_vec = np.nan * np.ones((num_wl_vec,))
        for i, xi in enumerate(workload_mat):
            if np.any(xi < - self.tol):
                draining_workload_vec[i] = self.is_draining_workload_vector(xi[:, None])
            else:
                draining_workload_vec[i] = 1
        assert is_binary(draining_workload_vec)
        return draining_workload_vec

    @staticmethod
    def return_draining_workload_vectors(workload_mat: types.WorkloadMatrix,
                                         draining_workload_vec: types.WorkloadSpace) \
            -> types.WorkloadMatrix:
        """
        Returns a subset of the workload matrix that contains only the valid rows indexed (with
        ones) by draining_workload_vec.

        :param workload_mat: Workload matrix, with rows being workload vectors.
        :param draining_workload_vec: Binary array with ones indexing valid workload vectors.
        :return draining_workload_matrix: New workload matrix, with a subset of the rows of the
            original workload matrix, which correspond with the valid (i.e., draining) ones.
        """
        draining_workload_matrix = workload_mat[draining_workload_vec == 1, :]
        return draining_workload_matrix
