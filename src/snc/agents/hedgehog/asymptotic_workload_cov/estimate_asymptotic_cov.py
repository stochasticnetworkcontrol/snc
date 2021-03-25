from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
from src import snc as types
from src.snc import AgentInterface
from src.snc.environments import ControlledRandomWalk
from src.snc import WorkloadTuple
from src.snc import SncSimulator


class EstimateAsymptoticWorkloadCovBatchMeans:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def generate_data_for_asymptotic_workload_cov_estimation(env: ControlledRandomWalk,
                                                             agent: AgentInterface,
                                                             num_data: int) \
            -> Dict[str, List[np.ndarray]]:
        """

        :param env: Environment to be simulated.
        :param agent: Agent used to generate data for the estimation.
        :param num_data: Number of data points.
        :return:
        """
        policy_simulator = SncSimulator(env, agent=agent, discount_factor=None)
        fake_data = policy_simulator.perform_online_simulation(num_simulation_steps=num_data)
        return fake_data

    @staticmethod
    def get_batch_sum(data: np.ndarray, batch_len: int, index: int) -> np.ndarray:
        """
        Returns the sum of a batch of data samples. This method is used for estimating the
        asymptotic covariance matrix with the "batch means method".

        :param data: matrix where each column is a data sample.
        :param batch_len: number of samples in the batch.
        :param index: batch index.
        :return: sum of samples in the batch.
        """
        batch = np.sum(data[:, index * batch_len: (index + 1) * batch_len], axis=1)
        return batch

    @staticmethod
    def get_noise_process(workload_mat: types.WorkloadMatrix, state_process: types.StateProcess,
                          action_process: types.ActionProcess, load: types.WorkloadSpace,
                          buffer_processing_matrix: types.BufferMatrix) -> types.WorkloadProcess:
        """
        Obtains noise process from data, i.e. state and action samples.

        :param workload_mat: workload matrix.
        :param state_process: matrix where the t-th column represents the state at time t.
        :param action_process: matrix where the t-th column represents the action at time t.
        :param load: vector with loads for every workload vector.
        :param buffer_processing_matrix: Matrix that defines the influence of each activity
            (column) in each buffer (row), and it is used in the transition dynamics.
        :return: noise_process: matrix with one column per sample of the noise process.

        """
        # W(t) = Xi * Q(t)
        workload_process = np.dot(workload_mat, state_process)

        # W(t+1) - W(t)
        diff_w = workload_process[:, 1:] - workload_process[:, 0:-1]

        # delta = 1 - rho
        drift = (1 - load).reshape((load.shape[0], 1))

        # I(t) = 1 + Xi * B * U(t)
        idleness_process = 1 + workload_mat @ buffer_processing_matrix @ action_process[:, 0:-1]

        # Delta(T+1) = W(t+1) - W(t) + delta - I(t)
        noise_process = diff_w + drift - idleness_process
        return noise_process

    @staticmethod
    def compute_workload_cov(noise_process: types.WorkloadProcess, batch_len: int) \
            -> types.WorkloadCov:
        """
        Computes the workload covariance from the noise process (from data) using batch means.

        :param noise_process: matrix with one column per sample of the noise process.
        :param batch_len: number of samples in the batch.
        :return: workload_cov: estimate of the asymptotic covariance matrix.
        """
        # Process stored in matrix form, each sample being a column vector.
        num_workload_vec, time_horizon = noise_process.shape

        num_batches = np.floor_divide(time_horizon, batch_len)

        # Sigma = 1/T * sum_{n=1}^N F^n_{T/N} * ( F^n_{T/N} )^T
        workload_cov = np.zeros((num_workload_vec, num_workload_vec))

        for n in range(int(num_batches)):
            # F^n_{len}, len=T/N
            batch = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(noise_process, batch_len,
                                                                          n)
            workload_cov += np.outer(batch, batch)

        workload_cov /= time_horizon
        return workload_cov

    @staticmethod
    def batch_means(fake_data, workload_tuple: WorkloadTuple, num_batch: int,
                    num_data: int, buffer_processing_matrix: types.BufferMatrix):
        """

        :param fake_data:
        :param workload_tuple:
        :param num_batch:
        :param num_data:
        :param buffer_processing_matrix:
        :return:
        """
        state_process = np.asarray(fake_data['state']).T
        action_process = np.asarray(fake_data['action']).T

        noise_process = EstimateAsymptoticWorkloadCovBatchMeans.get_noise_process(
            workload_tuple.workload_mat, state_process, action_process, workload_tuple.load,
            buffer_processing_matrix)

        batch_len = int(np.floor_divide(num_data, num_batch))
        assert batch_len > 0

        workload_cov = EstimateAsymptoticWorkloadCovBatchMeans.compute_workload_cov(noise_process,
                                                                                    batch_len)
        return workload_cov

    def estimate_asymptotic_workload_cov(self, buffer_processing_matrix: types.BufferMatrix,
                                         workload_tuple: WorkloadTuple,
                                         num_batch: int,
                                         num_data: int,
                                         agent: Optional[AgentInterface] = None,
                                         debug_info: bool = False) \
            -> types.WorkloadCov:
        """
        Estimate the asymptotic workload covariance from samples by running a simulator on the
        hedgehog policy for a certain number of time steps, with a basic assumption of a diagonal
        workload covariance.

        :param buffer_processing_matrix: Buffer processing matrix from environment.
        :param workload_tuple: The workload tuple for the system.
        :param num_batch: Number of batches to group data samples for covariance estimation.
        :param num_data: The number of simulated time steps for asymptotic cov. estimation.
        :param agent: Agent used to generate data for the estimation.
        :param debug_info: Boolean flag that indicates whether printing useful debug info.
        :return: workload_cov: Estimate of the asymptotic covariance.
        """
        fake_data = self.generate_data_for_asymptotic_workload_cov_estimation(
            self.env,
            agent,
            num_data
        )
        workload_cov = self.batch_means(
            fake_data,
            workload_tuple,
            num_batch,
            num_data,
            buffer_processing_matrix
        )
        if debug_info:
            print(f"workload_cov: {workload_cov}")
        return workload_cov

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        d = deepcopy(self.__dict__)
        return d
