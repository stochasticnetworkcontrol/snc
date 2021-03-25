import numpy as np
from src import snc as types
from src.snc.environments \
    import ScaledBernoulliServicesAndArrivalsGenerator
from src.snc \
    import ComputeAsymptoticCovBernoulliServiceInterface


class ComputeAsymptoticCovBernoulliServiceAndArrivals\
            (ComputeAsymptoticCovBernoulliServiceInterface):

    def __init__(self, job_generator: ScaledBernoulliServicesAndArrivalsGenerator,
                 constituency_matrix: types.ConstituencyMatrix,
                 workload_matrix: types.WorkloadMatrix,
                 tol: float = 1e-6):
        """

        :param job_generator: From environment.
        :param constituency_matrix: From environment.
        :param workload_matrix: Workload matrix.
        """
        assert isinstance(job_generator, ScaledBernoulliServicesAndArrivalsGenerator), ""\
            "Job generator must be an instance of " \
            "'ScaledBernoulliServicesPoissonArrivalsGenerator'."
        super().__init__(job_generator, constituency_matrix, workload_matrix, tol)

    @staticmethod
    def compute_covariance_arrival_process(demand_rate: types.StateSpace) -> np.ndarray:
        """
        Returns covariance of arrival process with mean given by 'demand_rate' under the assumption
        that it is a Bernoulli process.

        :param demand_rate: Mean of the random variables corresponding with instantaneous arrivals.
        :return: Covariance matrix of the arrival Poisson process.
        """
        return np.diag(np.squeeze(demand_rate * (1 - demand_rate)))