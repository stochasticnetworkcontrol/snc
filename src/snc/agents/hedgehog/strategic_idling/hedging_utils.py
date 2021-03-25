from typing import List

from src import snc as types
import numpy as np


def get_index_deficit_buffers(workload_mat: types.WorkloadMatrix, eps: float = 1e-7) -> List[int]:
    """
    Returns the indexes of the columns that correspond with deficit buffers. The workload matrix has
    both positive and negative entries for demand models. Deficit buffers can be identified from the
    workload matrix because they correspond with columns that have all nonnegative entries and at
    least one strictly positive entry.
    The returned list might not contain all demand buffers, but it will return those that are
    relevant for the task of shifting the workload positively.

    :param workload_mat: workload matrix.
    :param eps: minimum value to check strict positivity.
    :return: deficit_buffers: index of deficit buffers.
    """
    positive_entries = np.argwhere(workload_mat > eps)
    columns_with_positive_entries = np.unique(positive_entries[:, 1])
    deficit_buffers = []
    for d in columns_with_positive_entries:
        assert np.all(workload_mat[:, d] >= - eps)
        deficit_buffers.append(d)
    assert deficit_buffers  # List must be non-empty.
    return deficit_buffers
