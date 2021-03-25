import numpy as np
from src.snc import ComputeAsymptoticCovInterface
import src.snc.environments.examples as examples


def test_compute_steady_state_policy_double_reentrant_line():
    env = examples.double_reentrant_line_only_shared_resources_model()
    buffer_processing_matrix = env.job_generator.buffer_processing_matrix
    demand_rate = env.job_generator.demand_rate
    alpha1 = demand_rate[0]
    mu1 = - buffer_processing_matrix[0, 0]
    mu2 = - buffer_processing_matrix[1, 1]
    mu3 = - buffer_processing_matrix[2, 2]
    mu4 = - buffer_processing_matrix[3, 3]

    policy = ComputeAsymptoticCovInterface.compute_steady_state_policy(
        buffer_processing_matrix, demand_rate, env.constituency_matrix)
    true_policy = alpha1 / np.array([[mu1], [mu2], [mu3], [mu4]])
    np.testing.assert_almost_equal(policy, true_policy)
