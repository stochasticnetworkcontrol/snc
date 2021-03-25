import numpy as np
import snc.agents.agents_utils as utils


def test_assert_orthogonal_rows_true():
    matrix = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    assert utils.has_orthogonal_rows(matrix)


def test_assert_orthogonal_rows_false():
    matrix = np.array([[1, 0, 0, 1], [1, 1, 1, 0]])
    assert not utils.has_orthogonal_rows(matrix)
