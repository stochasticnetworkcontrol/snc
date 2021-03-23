import numpy as np


def is_binary(a: np.ndarray) -> bool:
    return bool(np.all(np.logical_or(a == 0, a == 1)))


def is_approx_binary(a: np.ndarray, eps=1e-5) -> bool:
    aa = np.abs(a)
    return bool(np.all(np.logical_and(
        a >= - eps, np.logical_or(aa <= eps, np.logical_and(aa >= 1 - eps, aa <= 1 + eps)))))


def is_binary_negative(a: np.ndarray) -> bool:
    return bool(np.all(np.logical_or(a == 0, a == -1)))


def is_routing_network(env) -> bool:
    drain_matrix = env.job_generator.draining_jobs_rate_matrix
    return bool(np.any(np.sum(drain_matrix < 0, axis=1) > 1))
