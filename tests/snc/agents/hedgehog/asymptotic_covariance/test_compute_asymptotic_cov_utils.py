import numpy as np
from src.snc \
    import EstimateAsymptoticWorkloadCovBatchMeans
from src.snc.agents.hedgehog.workload import workload
from src.snc import SteadyStatePolicyAgent


def perform_test(env, num_batch, num_data, class_kind, job_gen_seed):
    """
    This test is designed to be imported by compute asymptotic covariance classes.
    """
    workload_tuple = workload.compute_load_workload_matrix(env)
    workload_cov_computer = class_kind(env.job_generator, env.constituency_matrix,
                                       workload_tuple.workload_mat)
    workload_cov_estimator = EstimateAsymptoticWorkloadCovBatchMeans(env)

    workload_cov_com = workload_cov_computer.compute_asymptotic_workload_cov()
    agent = SteadyStatePolicyAgent(env, agent_seed=job_gen_seed + 200, mpc_seed=job_gen_seed + 300)
    workload_cov_est = workload_cov_estimator.estimate_asymptotic_workload_cov(
        env.job_generator.buffer_processing_matrix, workload_tuple, num_batch, num_data, agent)
    assert np.linalg.norm(workload_cov_est - workload_cov_com) / np.linalg.norm(workload_cov_com) \
           <= 0.2
