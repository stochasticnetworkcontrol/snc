import numpy as np
from abc import abstractmethod
from src.snc.environments \
    import  ScaledJobGeneratorInterface
from src import snc as snc_types, snc as snc_tools


class ScaledBernoulliServicesGeneratorInterface(ScaledJobGeneratorInterface):
    """
    Generates arrivals and services jobs according to independent stochastic processes. The number
    of service completions follow a Bernoulli distribution with mean equal to the mean rate scaled
    such that it is equal or smaller than one:
       max_mu = buffer_processing_matrix
       mu_j_new = mu_j / (max_mu + epsilon), for all j = 1, ..., ell_u,
    where epsilon is given by the add_max_rate parameter.
    The new arrivals process has to be implemented.
    """

    @abstractmethod
    def get_arrival_jobs(self) -> snc_types.StateSpace:
        """Generate integer number of new arrivals from some distribution with mean equal to
        the scaled mean rate."""
        pass

    def get_supplied_jobs(self, rate: float) -> np.int64:
        """Generate integer number of controlled arrivals at supply nodes from Bernoulli
        distribution with mean equal to the scaled mean rate."""
        assert rate >= 0
        assert rate <= 1
        return self.np_random.binomial(p=rate, n=1, size=1)

    def get_instantaneous_drained_jobs_matrix(self) -> snc_types.BufferMatrix:
        """Drain integer number of service completions from Bernoulli distribution with mean equal
        to the scaled mean rate at all non-supply nodes."""
        drain_mat_shape = self.buffer_processing_matrix.shape
        removed_services_matrix = - self.np_random.binomial(p=-self.draining_jobs_rate_matrix,
                                                            n=np.ones(drain_mat_shape, dtype=int),
                                                            size=drain_mat_shape)
        assert snc_tools.is_binary_negative(removed_services_matrix)
        return removed_services_matrix
