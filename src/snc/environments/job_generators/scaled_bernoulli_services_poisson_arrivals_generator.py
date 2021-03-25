from src.snc.environments \
    import ScaledBernoulliServicesGeneratorInterface
from src import snc as snc_types


class ScaledBernoulliServicesPoissonArrivalsGenerator(ScaledBernoulliServicesGeneratorInterface):
    """
    Generates arrivals and services jobs according to independent stochastic processes. The number
    of service completions follow a Bernoulli distribution with mean equal to the mean rate scaled
    such that it is equal or smaller than one:
       max_mu = buffer_processing_matrix
       mu_j_new = mu_j / (max_mu + epsilon), for all j = 1, ..., ell_u,
    where epsilon is given by the add_max_rate parameter.
    The number of new arrivals follows a Poisson random variable, with mean rate scaled by the
    same factor as the service completion rate:
       alpha_i_new = alpha_i / (max_mu + epsilon), for all i = 1, ..., ell.
    """

    def get_arrival_jobs(self) -> snc_types.StateSpace:
        """Generate integer number of new arrivals from Poisson distribution with mean equal to
        the scaled mean rate."""
        return self.np_random.poisson(self.demand_rate)
