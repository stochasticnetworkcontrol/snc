import numpy as np
import pytest
from snc.agents.hedgehog.workload import workload
from snc.agents.hedgehog.workload.validate_negative_workload_vectors import ValidateNegativeWorkloadVectors
import snc.environments.examples as examples


def compute_workload_and_draining_workload(env, tol):
    # Compute all workload vectors for o_s = 1.
    workload_mat = workload.compute_full_workload_matrix_velocity_projection_cdd(env)
    # Remove the non-draining negative workload vectors if any.
    draining_workload = ValidateNegativeWorkloadVectors(env.constituency_matrix,
                                                        env.job_generator.buffer_processing_matrix,
                                                        workload_mat, tol)
    draining_workload_vec = draining_workload.identify_draining_workload_vectors(workload_mat)
    workload_mat_drain = workload_mat[draining_workload_vec == 1, :]
    return workload_mat, workload_mat_drain


def assert_equal_to_main_workload_tuple_method(env, workload_mat, tol):
    workload_tuple = workload.compute_load_workload_matrix(env, tol=tol)
    workload_mat_s1 = np.sort(workload_mat, axis=0)
    workload_mat_s2 = np.sort(workload_tuple.workload_mat, axis=0)
    np.testing.assert_almost_equal(workload_mat_s1, workload_mat_s2)


@pytest.fixture(params=[
    examples.routing_with_negative_workload(
        alpha1=0.45, mu1=0.5, mu2=0.55, mu3=0.48),
    examples.simple_link_constrained_model(
        alpha1=3.4, mu12=2, mu13=5, mu25=2, mu32=4.5, mu34=1.8, mu35=2, mu45=1, mu5=7,
        cost_per_buffer=np.array([[1], [1], [3], [1.5], [3]]))
])
def models_negative_filling_workload_fix(request):
    return request.param


@pytest.fixture(params=[examples.simple_reentrant_line_model(),
                        examples.ksrs_network_model()])
def models_nonnegative_workload_fix(request):
    return request.param


@pytest.fixture(params=[examples.single_station_demand_model(),
                        examples.double_reentrant_line_with_demand_only_shared_resources_model()])
def pull_model_fix(request):
    return request.param


def test_models_with_negative_filling_workload_vectors(
        models_negative_filling_workload_fix):
    # Scenario under test with one negative workload vector component.
    env = models_negative_filling_workload_fix
    tol = 1e-4

    workload_mat, workload_mat_drain = compute_workload_and_draining_workload(env, tol)
    assert np.any(workload_mat < -tol)
    assert np.all(workload_mat_drain >= -tol)

    assert_equal_to_main_workload_tuple_method(env, workload_mat_drain, tol)


def test_minimal_routing_example_with_negative_entry_and_double_draining_action():
    # Scenario under test with one negative workload vector component.
    env = examples.routing_with_negative_workload_and_double_draining_action(
        alpha1=0.45, mu1=0.5, mu2=0.55, mu3=0.48)
    tol = 1e-4

    workload_mat, workload_mat_drain = compute_workload_and_draining_workload(env, tol)
    np.testing.assert_almost_equal(workload_mat, workload_mat_drain)
    assert np.any(workload_mat < -tol)
    assert np.any(workload_mat_drain < -tol)
    assert_equal_to_main_workload_tuple_method(env, workload_mat_drain, tol)


def test_push_models_with_only_nonnegative_entries_so_no_pruning(models_nonnegative_workload_fix):
    """
    No negative entries, so no change.
    """
    tol = 1e-4
    env = models_nonnegative_workload_fix

    workload_mat, workload_mat_drain = compute_workload_and_draining_workload(env, tol)
    np.testing.assert_almost_equal(workload_mat, workload_mat_drain)
    assert np.all(workload_mat >= - tol)
    assert_equal_to_main_workload_tuple_method(env, workload_mat_drain, tol)


def test_pull_models_with_draining_negative_entries_so_no_pruning(pull_model_fix):
    """
    Negative entries that drain, so no change.
    """
    tol = 1e-4
    env = pull_model_fix

    workload_mat, workload_mat_drain = compute_workload_and_draining_workload(env, tol)
    assert np.any(workload_mat < - tol)
    assert_equal_to_main_workload_tuple_method(env, workload_mat_drain, tol)
