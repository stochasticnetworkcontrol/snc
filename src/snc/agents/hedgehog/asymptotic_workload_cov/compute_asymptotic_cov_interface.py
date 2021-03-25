from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict

import numpy as np
import cvxpy as cvx
import snc.utils.snc_types as types
from snc.agents import agents_utils
from snc.environments.job_generators.job_generator_interface import \
    JobGeneratorInterface


class ComputeAsymptoticCovInterface(ABC):

    def __init__(self, job_generator: JobGeneratorInterface,
                 constituency_matrix: types.ConstituencyMatrix,
                 workload_matrix: types.WorkloadMatrix):
        """
        Interface to computes the asymptotic covariance in closed form. The children classes
        implement the methods for the supported job generators.

        :param job_generator: From environment.
        :param constituency_matrix: From environment. It must have orthogonal rows.
        :param workload_matrix: Workload matrix.
        """
        self._buffer_processing_matrix = job_generator.buffer_processing_matrix
        self._demand_rate = job_generator.demand_rate
        assert agents_utils.has_orthogonal_rows(constituency_matrix), \
            "Constituency matrix must have orthogonal rows."
        self._constituency_matrix = constituency_matrix
        self._workload_matrix = workload_matrix

    @staticmethod
    @abstractmethod
    def compute_covariance_arrival_process(demand_rate: types.StateSpace) -> np.ndarray:
        """
        Returns covariance of arrival process with mean given by 'demand_rate' under the assumption
        that it is a Martingale difference sequence.

        :param demand_rate: Mean of the random variables corresponding with instantaneous arrivals.
        :return: Covariance matrix of the arrival process.
        """

    @staticmethod
    @abstractmethod
    def compute_variance_single_entry_service_process(mu: float) -> float:
        """
        Returns variance of scalar random variable associated with an entry of the buffer processing
        matrix with mean 'mu'.

        :param mu: Mean of the scalar random variable.
        :return: Variance of the scalar random variable.
        """

    @staticmethod
    def compute_steady_state_policy(buffer_processing_matrix: types.BufferMatrix,
                                    demand_rate: types.StateSpace,
                                    constituency_matrix: types.ConstituencyMatrix) \
            -> types.ActionSpace:
        """
        Return steady state policy, z ≡ lim_{t→∞} E[U(t)], by solving the following problem::
            min     sum(z)
            s.t.    B z + alpha == 0.
                    C z <= 1
                    z >= 0.

        :param buffer_processing_matrix: From the environment.
        :param demand_rate: From the environment.
        :param constituency_matrix: From the environment.
        :return: Steady state policy E_∞[U(t)].
        """
        num_activities = buffer_processing_matrix.shape[1]
        z = cvx.Variable((num_activities, 1))
        objective = cvx.Minimize(cvx.sum(z))
        constraints = [
            buffer_processing_matrix @ z + demand_rate == 0,
            constituency_matrix @ z <= 1,
            z >= 0]
        cvx.Problem(objective, constraints).solve(solver="CPLEX")
        assert z.value is not None, f"Problem is infeasible. Check that network load < 1."
        return z.value

    def compute_asymptotic_cov_service_process(self) -> np.ndarray:
        """
        Return the asymptotic covariance of the service process under the following assumptions:
        i) The columns of the buffer processing matrix are independent random variables.
        ii) The nonzero entries of each column of the buffer processing matrix are perfectly
            correlated: they are scaled versions of each other.
        iii) The process is a Martingale difference sequence.

        :return: Asymptotic covariance of the service process.
        """
        action_steady_state = self.compute_steady_state_policy(
            self._buffer_processing_matrix, self._demand_rate, self._constituency_matrix)

        num_buffers, num_actions = self._buffer_processing_matrix.shape
        cov_service = np.zeros((num_buffers, num_buffers))  # Initialise sum of each column cov.

        for j in range(num_actions):  # For each column.
            # j-th column of buffer_processing_matrix.
            b = self._buffer_processing_matrix[:, j][:, None]

            # Row index with maximum absolute value in j-th column of buffer_processing_matrix.
            ind_max_rate_j = np.argmax(np.abs(b))
            # Use the service rate as a probability measure.
            p = np.abs(b[ind_max_rate_j])
            # Compute variance of the maximum value of the j-th column.
            var_b_max_j = self.compute_variance_single_entry_service_process(p)

            # Scaling factor for all entries of j-th column with respect to its maximum value.
            a = b / b[ind_max_rate_j]
            # Covariance of the j-th column.
            cov_bj = var_b_max_j * (a @ a.T)

            cov_service += cov_bj * action_steady_state[j, 0]  # Add to covariance of other columns.
        return cov_service

    def compute_asymptotic_workload_cov(self) -> np.ndarray:
        """
        Computes the asymptotic covariance of the workload process for the CRW, which assumes i.i.d.
        arrivals and service processes.

        :return: Asymptotic covariance matrix.
        """
        cov_demand = self.compute_covariance_arrival_process(self._demand_rate)
        cov_service = self.compute_asymptotic_cov_service_process()
        cov_state_process = cov_demand + cov_service
        cov_workload_matrix = self._workload_matrix @ cov_state_process @ self._workload_matrix.T
        return cov_workload_matrix

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        d = deepcopy(self.__dict__)
        return d
