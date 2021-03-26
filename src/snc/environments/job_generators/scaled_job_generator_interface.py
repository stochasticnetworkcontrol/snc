from abc import abstractmethod
from typing import Optional
import numpy as np
from snc.environments.job_generators.job_generator_interface \
    import JobGeneratorInterface
import snc.utils.snc_types as snc_types


class ScaledJobGeneratorInterface(JobGeneratorInterface):

    def __init__(self,
                 demand_rate: snc_types.StateSpace,
                 buffer_processing_matrix: snc_types.BufferMatrix,
                 job_gen_seed: Optional[int] = None,
                 add_max_rate: float = 1e-1) -> None:
        """
        Generates and services jobs according to independent stochastic processes. The number of
        arrivals and service completions will follow distributions with mean equal to the mean rate
        scaled such that it is equal or smaller than one:
            max_mu = buffer_processing_matrix
            mu_j_new = mu_j / (max_mu + epsilon), for all j = 1, ..., ell_u,
            alpha_i_new = alpha_i / (max_mu + epsilon), for all i = 1, ..., ell.
        where epsilon is given by the `add_max_rate` parameter.
        The methods for generating service completions and arrivals have to be implemented.

        :param demand_rate: Column vector with demand rate at every buffer, alpha.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :param job_gen_seed: The number to initialise the numpy RandomState.
        :param add_max_rate: Additional percentage scaling factor for the mean rates. If
            add_max_rate = 0, then the activity with highest mean rate will generate a service
            completion with probability one.
        """
        assert 0 <= add_max_rate <= 1
        self.add_max_rate = add_max_rate
        self.max_mean_rate = np.max(np.abs(buffer_processing_matrix))
        self.sim_time_interval = self.compute_sim_time_interval()

        # Store original (unscaled) buffer processing matrix and compute its scaled version.
        self.unscaled_buffer_processing_matrix = buffer_processing_matrix
        scaled_buffer_processing_matrix = buffer_processing_matrix * self.sim_time_interval

        # Store original (unscaled) demand rates vector and compute its scaled version.
        self.unscaled_demand_rate = demand_rate
        scaled_demand_rate = demand_rate * self.sim_time_interval

        # Create a job generator with the scaled rates.
        super().__init__(scaled_demand_rate, scaled_buffer_processing_matrix, job_gen_seed)

    def compute_sim_time_interval(self):
        """
        If time interval to scale time-steps is not given as a parameter, this function computes
        a time interval that makes the mean service rate smaller or equal than one."""
        if self.max_mean_rate > 1:
            return 1 / (self.max_mean_rate * (1 + self.add_max_rate))
        else:
            return 1

    @abstractmethod
    def get_arrival_jobs(self) -> snc_types.StateSpace:
        """Returns number of new job arrivals at each buffer."""
        pass

    @abstractmethod
    def get_instantaneous_drained_jobs_matrix(self) -> snc_types.BufferMatrix:
        """
        Returns the number of processed jobs with given rate (take absolute value to make rate
        positive). The entries are negative to indicate that they leave the buffer.
        """
        pass

    @abstractmethod
    def get_supplied_jobs(self, rate: float) -> np.int64:
        """
        Returns the number of supplied jobs with given rate. The entries are positive to indicate
        that they add to the buffer.
        """
        pass
