import numpy as np
import snc.utils.snc_types as types
from snc.environments.job_generators.\
    scaled_bernoulli_services_poisson_arrivals_generator \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from snc.agents.hedgehog.asymptotic_workload_cov.\
    compute_asymptotic_cov_bernoulli_service_interface \
    import ComputeAsymptoticCovBernoulliServiceInterface


class ComputeAsymptoticCovBernoulliServicePoissonArrivals\
            (ComputeAsymptoticCovBernoulliServiceInterface):

    def __init__(self, job_generator: ScaledBernoulliServicesPoissonArrivalsGenerator,
                 constituency_matrix: types.ConstituencyMatrix,
                 workload_matrix: types.WorkloadMatrix,
                 tol: float = 1e-6):
        """

        :param job_generator: From environment.
        :param constituency_matrix: From environment.
        :param workload_matrix: Workload matrix.
        """
        assert isinstance(job_generator, ScaledBernoulliServicesPoissonArrivalsGenerator), ""\
            "Job generator must be an instance of " \
            "'ScaledBernoulliServicesPoissonArrivalsGenerator'."
        super().__init__(job_generator, constituency_matrix, workload_matrix, tol)

    @staticmethod
    def compute_covariance_arrival_process(demand_rate: types.StateSpace) -> np.ndarray:
        """
        Returns covariance of arrival process with mean given by 'demand_rate' under the assumption
        that it is a Poisson point process.

        :param demand_rate: Mean of the random variables corresponding with instantaneous arrivals.
        :return: Covariance matrix of the arrival Poisson process.
        """
        return np.diag(np.squeeze(demand_rate))
