from typing import Optional, Tuple
import numpy as np
import snc.utils.snc_types as types
from snc.environments.job_generators. \
    scaled_bernoulli_services_generator_interface import ScaledBernoulliServicesGeneratorInterface


class ScaledBernoulliServicesGeometricXBernoulliArrivalsGenerator\
            (ScaledBernoulliServicesGeneratorInterface):

    def __init__(self, demand_rate: types.StateSpace,
                 demand_variance: types.StateSpace,
                 buffer_processing_matrix: types.BufferMatrix,
                 job_gen_seed: Optional[int] = None,
                 add_max_rate: Optional[float] = 1e-1) -> None:
        """
        Generates and process jobs according to independent stochastic processes, where at each
        time step, the number of new arrivals and follow a distribution that is
        the product of a Bernoulli and a geometric distributions; and the number of serviced jobs
        follows a Bernoulli distribution.
        The mean of each process is scaled such that it is equal or smaller than one:
            max_mu = buffer_processing_matrix + add_max_rate
            mu_j_new = mu_j / (max_mu + epsilon), for all j = 1, ..., ell_u.
            alpha_i_new = alpha_i / (max_mu + epsilon), for all i = 1, ..., ell.
        The parameters of the Bernoulli and geometric distributions for the arrivals process are
        computed internally such that the resulting distribution matches the mean and variance given
        by `demand_mean` and `demand_variance` input parameters, respectively.

        :param demand_rate: Column vector with demand rate at every buffer, alpha.
        :param demand_variance: Array with variance of arrivals at every buffer.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :param job_gen_seed: The number to initialise the numpy RandomState.
        :param add_max_rate: Additional scaling factor for the mean rates. If epsilon = 0, then the
            activity with highest mean rate will generate a service completion with probability one.
        """
        self.job_gen_seed = job_gen_seed

        assert add_max_rate >= 0
        self._add_max_rate = add_max_rate

        super().__init__(demand_rate, buffer_processing_matrix, job_gen_seed, add_max_rate)

        assert np.all(demand_variance >= 0), "Demand variance must be nonnegative."
        assert np.all(demand_variance.shape == self.demand_rate.shape), \
            "Demand variance and mean parameters must have the same shape."
        assert np.all(demand_variance >= self.demand_rate - self.demand_rate ** 2)
        self.demand_variance = demand_variance

        # Get parameters of geometric and Bernoulli dist, whose product give the demand dist.
        self.demand_geom_p, self.demand_bern_p = self.get_params(self.demand_rate, demand_variance)

    @staticmethod
    def get_params(mean: types.StateSpace, variance: types.StateSpace) \
            -> Tuple[types.StateSpace, types.StateSpace]:
        """
        Returns the parameters of two distributions: a Bernoulli and a Geometric distributions, such
        that their product, A(t) = Ber(t) * (1 + G(t)), has the first and second order moment given
        by the input parameters `mean` and `variance`, respectively.

        :param mean: First moment of every random variable of the i.i.d. process.
        :param variance: Second moment of every random variable of the i.i.d. process.
        :return: (geometric_p, bernoulli_p).
            - geometric_p: Parameter of Geometric distribution.
            - bernoulli_p: Parameter of Bernoulli distribution.
        """

        geometric_p = 2 * mean / (variance + mean + mean ** 2)
        bernoulli_p = mean * geometric_p
        return geometric_p, bernoulli_p

    def get_arrival_jobs(self) -> types.StateSpace:
        """
        Generate an integer number of new arrivals from a random variable that is the product of a
        Bernoulli times a geometric random variables:
            A(t) = Ber(t) * (1 + G(t)),
        where
            P{Ber(t) = 1} = rho, and P{Ber(t) = 0} = 1 - rho,
        and
            P{G(t) >= 0} = (1 - gamma)^k, k >= 0.
        Note that we define the geometric distribution as the distribution of the number Y = X -1 of
        failures before the first success. However, Numpy defines the geometric distribution as the
        distribution of the number X of Bernoulli trials needed to get one success:
            f(k) = (1 - p)^{k - 1} p, k >= 1.
        Hence, we have to convert our parameter gamma to the Numpy parameter p:
            p = gamma / (1 - gamma).

        :return: Sample column vector from the arrival process.
        """
        new_arrivals = np.zeros((self.num_buffers, 1))

        ind = np.logical_not(np.isnan(self.demand_bern_p))
        arrival_event = self.np_random.binomial(p=self.demand_bern_p[ind], n=1)
        num_arrivals = self.np_random.geometric(p=self.demand_geom_p[ind])
        new_arrivals[ind] = arrival_event * num_arrivals
        return new_arrivals
