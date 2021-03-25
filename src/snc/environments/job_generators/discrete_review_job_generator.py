from typing import Optional
import numpy as np
from snc.environments.job_generators.job_generator_interface \
    import JobGeneratorInterface
import snc.utils.snc_types as snc_types


class DiscreteReviewJobGenerator(JobGeneratorInterface):

    def __init__(self,
                 demand_rate: snc_types.StateSpace,
                 buffer_processing_matrix: snc_types.BufferMatrix,
                 job_gen_seed: Optional[int] = None,
                 add_max_rate: float = 1e-1,
                 sim_time_interval: Optional[float] = None):
        """
        The number of service completion and new arrivals at every time-step follow some
        distribution (deterministic, Poisson, etc.). The mean rate of the distribution is scaled
        by parameter `time_interval`, which represents the time-elapse between consecutive
        time-steps.

        :param demand_rate: Array with demand rate at every buffer, alpha.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :param job_gen_seed: The number to initialise the numpy RandomState.
        :param add_max_rate: Additional scaling factor for the mean rates. If epsilon = 0, then the
            activity with highest mean rate will generate a service completion with probability one.
        :param sim_time_interval: Simulated (continuous) time-elapse between consecutive (discrete)
            time-steps.
        """
        if sim_time_interval:
            assert sim_time_interval > 0
            self.sim_time_interval = sim_time_interval
        else:
            assert add_max_rate >= 0
            self.add_max_rate = add_max_rate
            self.max_mean_rate = np.max(np.abs(buffer_processing_matrix))
            self.sim_time_interval = self.compute_sim_time_interval()

        self.unscaled_buffer_processing_matrix = buffer_processing_matrix
        super().__init__(demand_rate, buffer_processing_matrix, job_gen_seed)

    def compute_sim_time_interval(self):
        """
        If time interval to scale time-steps is not given as a parameter, this function computes
        a time interval that makes the mean service rate smaller or equal than one."""
        if self.max_mean_rate > 1:
            return 1 / (self.max_mean_rate * (1 + self.add_max_rate))
        else:
            return 1

    def get_arrival_jobs(self):
        pass

    def get_instantaneous_drained_jobs_matrix(self):
        pass

    def get_supplied_jobs(self, rate: float) -> np.int64:
        pass


class DeterministicDiscreteReviewJobGenerator(DiscreteReviewJobGenerator):
    """
    Generates and process jobs deterministically at a given rate.
    """
    def get_arrival_jobs(self):
        return self.demand_rate * self.sim_time_interval

    def get_instantaneous_drained_jobs_matrix(self):
        return self.draining_jobs_rate_matrix * self.sim_time_interval

    def get_supplied_jobs(self, rate: float) -> np.int64:
        assert rate >= 0
        return np.round(rate * self.sim_time_interval)


class PoissonDiscreteReviewJobGenerator(DiscreteReviewJobGenerator):
    """
    Generates and process jobs according to independent stochastic arrival and service
    completion processes, where the number of jobs at each time-step follows a Poisson
    distribution with some given mean rate.
    Note: This is different from the CRW described in the book, where the number of service
    completions at each time step is in {0, 1}, such that the entries of B(t+1) are in
    {-1, 0, 1}.
    In comparison, the entries of B(t+1) returned by this method could be any integer if the
    specified mean rate is large enough. However, if time_interval is small enough (i.e.
    time_interval << max mean service rate), then we can expect that this generator will
    approximate a Poisson point process well.
    """
    def get_arrival_jobs(self) -> snc_types.StateSpace:
        return self.np_random.poisson(self.demand_rate * self.sim_time_interval)

    def get_instantaneous_drained_jobs_matrix(self) -> snc_types.BufferMatrix:
        return - (self.np_random.poisson(np.abs(self.draining_jobs_rate_matrix)
                                         * self.sim_time_interval))

    def get_supplied_jobs(self, rate: float) -> np.int64:
        assert rate >= 0
        return self.np_random.poisson(rate * self.sim_time_interval)
