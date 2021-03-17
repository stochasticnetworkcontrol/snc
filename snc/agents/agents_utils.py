import numpy as np
import snc.utils.snc_types as types


def has_orthogonal_rows(binary_matrix: types.Matrix) -> bool:
    """
    Ensure that matrix has binary_matrix has orthogonal rows, i.e., there is one and only one
    unit value per column.

    :param binary_matrix:
    :return: True if binary_matrix is orthogonal.
    """
    num_columns = binary_matrix.shape[1]
    # casting to bool to satisfy mypy disagreement with type np.bool_
    return bool(np.all(np.sum(binary_matrix, axis=0) == np.ones((num_columns,))))
