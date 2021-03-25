import numpy as np
from src import snc as utils


def test_assert_orthogonal_rows_true():
    matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    assert utils.has_orthogonal_rows(matrix)


def test_assert_orthogonal_rows_false():
    matrix = np.array([[1, 0, 0, 1], [1, 1, 1, 0]])
    assert not utils.has_orthogonal_rows(matrix)
