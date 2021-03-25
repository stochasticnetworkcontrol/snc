import numpy as np
from src.snc import EstimateAsymptoticWorkloadCovBatchMeans


def test_get_batch():
    data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]])

    b12 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=2, index=0)
    assert np.all(b12 == np.sum(np.array([[1, 2], [.1, .2]]), axis=1))
    b22 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=2, index=1)
    assert np.all(b22 == np.sum(np.array([[3, 4], [.3, .4]]), axis=1))
    b32 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=2, index=2)
    assert np.all(b32 == np.sum(np.array([[5, 6], [.5, .6]]), axis=1))
    b42 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=2, index=3)
    assert np.all(b42 == np.sum(np.array([[7, 8], [.7, .8]]), axis=1))
    b52 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=2, index=4)
    assert np.all(b52 == np.sum(np.array([[9, 10], [.9, 1]]), axis=1))

    b13 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=3, index=0)
    assert np.all(b13 == np.sum(np.array([[1, 2, 3], [.1, .2, .3]]), axis=1))
    b23 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=3, index=1)
    assert np.all(b23 == np.sum(np.array([[4, 5, 6], [.4, .5, .6]]), axis=1))
    b33 = EstimateAsymptoticWorkloadCovBatchMeans.get_batch_sum(data, batch_len=3, index=2)
    assert np.all(b33 == np.sum(np.array([[7, 8, 9], [.7, .8, .9]]), axis=1))


def test_get_noise_process_single_server():
    state_process = np.array([[0, 2, 3, 6, 4, 2, 2]])
    action_process = np.array([[1, 0, 0, 0, 1, 1, 1]])
    load = np.array([0.9])
    workload_mat = 0.5 * np.ones((1, 1))
    buffer_processing_matrix = -2 * np.ones((1, 1))

    test_noise = np.array([[1.1, -0.4, 0.6, -1.9, -0.9, 0.1]])

    noise_process = EstimateAsymptoticWorkloadCovBatchMeans.get_noise_process(
        workload_mat, state_process, action_process, load, buffer_processing_matrix)

    np.testing.assert_almost_equal(noise_process, test_noise)


def test_get_noise_process_two_serially_connected_resources_one_buffer_each():
    """We test multiple dimension processes: 2 buffers, 2 actions."""
    state_process = np.array([[0, 2, 3, 6, 4, 2, 2],
                              [1, 3, 4, 7, 5, 3, 3]])
    action_process = np.array([[1, 0, 0, 0, 1, 1, 1],
                               [0, 1, 1, 0, 1, 0, 1]])
    load = np.array([[0.9], [0.9]])
    workload_mat = np.array([[0.5, 0.5],
                             [0.1, 0.1]])
    buffer_processing_matrix = np.array([[-1.5, 0],
                                         [1.5, -0.5]])  # mu1 = 1.5, mu2 = 0.5

    test_noise = np.array([[1.1, 0.35, 2.35, -2.9, -2.65, -0.9],
                           [-0.5, -0.65, -0.25, -1.3, -1.25, -0.9]])

    noise_process = EstimateAsymptoticWorkloadCovBatchMeans.get_noise_process(
        workload_mat, state_process, action_process, load, buffer_processing_matrix)

    np.testing.assert_almost_equal(noise_process, test_noise)


def test_compute_workload_cov():
    """Simple correctness test."""
    noise = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]])
    b12 = np.sum(np.array([[1, 2], [.1, .2]]), axis=1)
    b22 = np.sum(np.array([[3, 4], [.3, .4]]), axis=1)
    b32 = np.sum(np.array([[5, 6], [.5, .6]]), axis=1)
    b42 = np.sum(np.array([[7, 8], [.7, .8]]), axis=1)
    b52 = np.sum(np.array([[9, 10], [.9, 1]]), axis=1)
    test_cov = np.outer(b12, b12) + np.outer(b22, b22) + np.outer(b32, b32) + np.outer(b42, b42) \
        + np.outer(b52, b52)
    test_cov /= 10
    workload_cov = EstimateAsymptoticWorkloadCovBatchMeans.compute_workload_cov(noise, 2)
    np.testing.assert_almost_equal(workload_cov, test_cov)
