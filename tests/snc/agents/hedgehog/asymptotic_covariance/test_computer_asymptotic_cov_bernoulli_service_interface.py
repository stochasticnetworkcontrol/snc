import numpy as np
import pytest

from snc.agents.hedgehog.asymptotic_workload_cov.\
    compute_asymptotic_cov_bernoulli_service_and_arrivals \
    import ComputeAsymptoticCovBernoulliServiceAndArrivals
from snc.agents.hedgehog.asymptotic_workload_cov.\
    compute_asymptotic_cov_bernoulli_service_poisson_arrivals \
    import ComputeAsymptoticCovBernoulliServicePoissonArrivals
import snc.agents.hedgehog.workload.workload as workload
from snc.environments import examples


@pytest.fixture(params=[ComputeAsymptoticCovBernoulliServicePoissonArrivals,
                        ComputeAsymptoticCovBernoulliServiceAndArrivals])
def class_fixture(request):
    return request.param


def test_compute_variance_single_entry_service_process_one(class_fixture):
    p = 1
    assert class_fixture.compute_variance_single_entry_service_process(p) == 0


def test_compute_variance_single_entry_service_process_zero(class_fixture):
    p = 0
    assert class_fixture.compute_variance_single_entry_service_process(p) == 0


def test_compute_variance_single_entry_service_process(class_fixture):
    p = 0.5
    assert class_fixture.compute_variance_single_entry_service_process(p) == 0.25


def test_compute_asymptotic_cov_service_process_non_orthogonal_constituency_matrix(class_fixture):
    env = examples.double_reentrant_line_only_shared_resources_model(initial_state=np.zeros((4, 1)))
    workload_tuple = workload.compute_load_workload_matrix(env)
    env.constituency_matrix[0, 1] = 1
    with pytest.raises(AssertionError):
        _ = class_fixture(env.job_generator, env.constituency_matrix, workload_tuple.workload_mat)
