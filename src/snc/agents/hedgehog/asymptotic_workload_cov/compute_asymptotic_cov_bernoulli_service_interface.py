from abc import abstractmethod

import numpy as np
from src import snc as types
from src.snc.environments import ScaledBernoulliServicesGeneratorInterface
from src.snc import ComputeAsymptoticCovInterface


class ComputeAsymptoticCovBernoulliServiceInterface(ComputeAsymptoticCovInterface):

    def __init__(self, job_generator: ScaledBernoulliServicesGeneratorInterface,
                 constituency_matrix: types.ConstituencyMatrix,
                 workload_matrix: types.WorkloadMatrix,
                 tol: float = 1e-6):
        """

        :param job_generator: From environment.
        :param constituency_matrix: From environment.
        :param workload_matrix: Workload matrix.
        """
        assert isinstance(job_generator, ScaledBernoulliServicesGeneratorInterface), ""\
            "Job generator must be an instance of " \
            "'ScaledBernoulliServicesGeneratorInterface'."
        assert np.all(- (1 + tol) <= job_generator.buffer_processing_matrix)
        assert np.all(job_generator.buffer_processing_matrix <= 1 + tol)
        super().__init__(job_generator, constituency_matrix, workload_matrix)

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
    def compute_variance_single_entry_service_process(p: float):
        """
        Returns variance of single entry of the buffer processing matrix under the assumption that
        it corresponds with the mean of a Bernoulli random variable.

        :param p: Mean of the Bernoulli random variable.
        :return: Variance of the Bernoulli random variable.
        """
        return p * (1 - p)
