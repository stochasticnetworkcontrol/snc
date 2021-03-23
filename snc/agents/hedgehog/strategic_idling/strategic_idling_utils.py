import numpy as np
import snc.utils.snc_types as types
from typing import Set


def is_pull_model(model_type: str) -> bool:
    """
    Function determines whether network is a pull model by checking the model type string.
    """
    return model_type == 'pull'


def get_dynamic_bottlenecks(w: types.WorkloadSpace,
                            workload_mat: types.WorkloadMatrix,
                            load: types.WorkloadSpace,
                            rel_tol: float = 0.05) -> Set[int]:
    """
    Method finds the set of bottlenecks which determine the minimum draining time
    in the network for the current workload.

    :param w: Current state in workload space, i.e. w = Xi x.
    :param workload_mat: Workload matrix (Xi in the equation w = Xi x).
    :param load: Load vector.
    :param rel_tol: Relative tolerance when comparing draining times of bottlenecks
    :return: Set of dynamic bottleneck indices
    """
    min_drain_time = 0
    dynamic_bottlenecks: Set[int] = set()
    # Use time that it takes to process one item from the slowest buffer
    # as a robust proxy for discretisation error in minimum draining time.
    abs_tols = np.max(np.abs(workload_mat), axis=1, keepdims=True) / (1 - load)
    max_abs_tol = np.max(abs_tols)
    for i, (w_i, load_i) in enumerate(zip(w, load)):
        drain_time = w_i / (1 - load_i)
        tol = max(rel_tol * max(min_drain_time, drain_time), max_abs_tol)
        if drain_time > min_drain_time + tol:
            # In this case we reset the dynamic bottlenecks since significantly
            # higher draining time has been found.
            dynamic_bottlenecks = set([i])
            min_drain_time = drain_time
        elif drain_time > min_drain_time - tol:
            # in this case several bottlenecks have identical draining time up
            # to a specified level of tolerance and hence we keep all of them
            # as dynamic bottlenecks.
            dynamic_bottlenecks.add(i)
            min_drain_time = max(min_drain_time, drain_time)

    return dynamic_bottlenecks
