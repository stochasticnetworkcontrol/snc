import numpy as np
import pytest

from snc.environments.job_generators.discrete_review_job_generator \
    import DeterministicDiscreteReviewJobGenerator as drjg
from snc.environments.job_generators.discrete_review_job_generator \
    import PoissonDiscreteReviewJobGenerator as prjg
from snc.environments.controlled_random_walk import ControlledRandomWalk
import snc.utils.snc_tools as snc
import snc.environments.state_initialiser as stinit
import snc.environments.examples as examples
import \
    snc.environments.examples_distribution_with_rebalancing as examples_distribution_with_rebalancing
from snc.environments.job_generators.scaled_bernoulli_services_poisson_arrivals_generator import \
    ScaledBernoulliServicesPoissonArrivalsGenerator
from snc.environments.state_initialiser import DeterministicCRWStateInitialiser


def test_is_binary():
    c = np.ones((1, 1))
    assert (snc.is_binary(c))

    c = np.zeros((1, 1))
    assert (snc.is_binary(c))

    c = np.ones((5, 4))
    assert (snc.is_binary(c))

    c = np.zeros((5, 4))
    assert (snc.is_binary(c))

    c = np.random.randint(0, 1, (3, 6))
    assert (snc.is_binary(c))

    c = []
    assert (not snc.is_binary(c))

    c = np.random.random_sample((3, 5))
    c[0] = 1
    assert (not snc.is_binary(c))


def test_index_phys_resources_with_negative_values():
    index_phys_resources = (-1, 0)

    # Other needed parameters
    cost_per_buffer = np.zeros((2, 1))
    demand_rate = np.zeros((2, 1))
    initial_state = np.zeros((2, 1))
    capacity = np.zeros((2, 1))
    constituency_mat = np.eye(2)
    buffer_processing_mat = np.eye(2)
    job_generator = drjg(demand_rate, buffer_processing_mat, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    with pytest.raises(AssertionError):
        ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                             index_phys_resources=index_phys_resources)


def test_index_phys_resources_with_index_higher_than_num_resources():
    index_phys_resources = (0, 2)

    # Other needed parameters
    cost_per_buffer = np.zeros((2, 1))
    demand_rate = np.zeros((2, 1))
    initial_state = np.zeros((2, 1))
    capacity = np.zeros((2, 1))
    constituency_mat = np.eye(2)
    buffer_processing_mat = np.eye(2)
    job_generator = drjg(demand_rate, buffer_processing_mat, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    with pytest.raises(AssertionError):
        ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                             index_phys_resources=index_phys_resources)


def test_index_phys_resources_with_repeated_indexes():
    index_phys_resources = (0, 0)

    # Other needed parameters
    cost_per_buffer = np.zeros((2, 1))
    demand_rate = np.zeros((2, 1))
    initial_state = np.zeros((2, 1))
    capacity = np.zeros((2, 1))
    constituency_mat = np.eye(2)
    buffer_processing_mat = np.eye(2)
    job_generator = drjg(demand_rate, buffer_processing_mat, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    with pytest.raises(AssertionError):
        ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                             index_phys_resources=index_phys_resources)


def test_valid_index_phys_resources_1():
    index_phys_resources = (0,)

    # Other needed parameters
    cost_per_buffer = np.zeros((2, 1))
    demand_rate = np.zeros((2, 1))
    initial_state = np.zeros((2, 1))
    capacity = np.zeros((2, 1))
    constituency_mat = np.eye(2)
    buffer_processing_mat = np.eye(2)
    job_generator = drjg(demand_rate, buffer_processing_mat, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                               index_phys_resources=index_phys_resources)
    assert env.index_phys_resources == index_phys_resources


def test_valid_index_phys_resources_1_2():
    index_phys_resources = (0, 1)

    # Other needed parameters
    cost_per_buffer = np.zeros((2, 1))
    demand_rate = np.zeros((2, 1))
    initial_state = np.zeros((2, 1))
    capacity = np.zeros((2, 1))
    constituency_mat = np.eye(2)
    buffer_processing_mat = np.eye(2)
    job_generator = drjg(demand_rate, buffer_processing_mat, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                               index_phys_resources=index_phys_resources)
    assert env.index_phys_resources == index_phys_resources


def test_state_initialiser_uniform():
    num_buffers = 5
    capacity = 10
    s0 = stinit.UniformRandomCRWStateInitialiser(num_buffers, capacity)
    init_state = np.zeros((num_buffers, 1))
    num_samples = 100000
    for i in range(num_samples):
        init_state += s0.get_initial_state()
    init_state /= num_samples
    np.all(np.isclose(init_state, ((capacity - 1) / 2) * np.ones((num_buffers, 1))))


def test_state_initialiser_deterministic():
    initial_state = np.array([2, 3, 4, 5])[:, None]
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)
    assert np.all(s0.get_initial_state() == initial_state)


def test_unfeasible_action():
    """Check that the step method doesn't allow actions that violate the one action per resource
    constraint: C u <= 1."""
    np.random.seed(42)
    env = examples.double_reentrant_line_only_shared_resources_model(alpha=0)
    env.reset_with_random_state(42)
    action = np.array([[1], [0], [0], [1]])
    with pytest.raises(AssertionError):
        _, _, _, _ = env.step(action)


def test_scheduling_single_buffer_events_constant():
    """One resource (station) with 1 buffers. Thus, there are 2 possible actions namely process or
    idle. At every iteration, we fill but also process the buffer, so the num of jobs remain
    constant = initial_state."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[1]])
    initial_state = np.array([[2]])
    capacity = np.array([[20]])
    job_generator = drjg(demand_rate, - np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    for i in range(100):
        s, r, d, t = env.step(action)
        assert s[0, 0] == initial_state
        assert r == - cost_per_buffer[0] * s


def test_scheduling_single_buffer_events_grow_one():
    """One resource (station) with 1 buffers. Thus, there are 2 possible actions namely process or
    idle. At every iteration, we process one job but fill with two, so jobs grow 1 at a time up to
    achieving maximum capacity."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[2]])
    initial_state = np.array([[2]])
    capacity = np.array([[20]])

    job_generator = drjg(demand_rate, - np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    for i in range(18):
        s, r, d, t = env.step(action)
        assert s[0, 0] == 1 + i + initial_state[0]
        assert r == - cost_per_buffer[0] * s
    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_scheduling_single_buffer_events_remove_two_until_empty():
    """One resource (station) with 1 buffers. Thus, there are 2 possible actions namely process or
    idle. We don't fill the buffer, just remove jobs, two at a time."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[0]])
    initial_state = np.array([[20]])
    capacity = np.array([[20]])
    job_generator = drjg(demand_rate, - 2 * np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    for i in range(10):
        s, r, d, t = env.step(action)
        assert s[0, 0] == np.max([0, initial_state[0] - (i + 1) * 2])
        assert r == - cost_per_buffer[0] * s
    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_scheduling_single_buffer_events_fill_one_but_remove_two_until_empty():
    """One resource (station) with 1 buffers. Thus, there are 2 possible actions namely process or
    idle. We fill the buffer with 1 job but remove two jobs per iteration, so it will decrease to
    zero."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[1]])
    initial_state = np.array([[20]])
    capacity = np.array([[20]])

    job_generator = drjg(demand_rate, - 2 * np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    for i in range(20):
        s, r, d, t = env.step(action)
        assert s[0, 0] == np.max([0, initial_state[0] - (i + 1)])
        assert r == - cost_per_buffer[0] * s
    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_scheduling_multiple_buffers_events():
    """One resource (station) with 3 buffers. Thus, there are 4 possible actions namely schedule
    each buffer or idle. At every iteration, if the resource chooses to schedule to work one buffer,
    then the jobs of that buffer are removed deterministically at some processing rate. Then, new
    jobs arrive also at some deterministic rate. The number of jobs in a buffer is always
    nonnegative and less than or equal to capacity. """
    cost_per_buffer = np.array([[1.1], [2.2], [3.3]])
    d1 = 1  # Rate of job arrival at buffer 1
    d2 = 2  # Rate of job arrival at buffer 2
    d3 = 3  # Rate of job arrival at buffer 3
    demand_rate = np.array([[d1], [d2], [d3]])
    initial_state = np.array([[0], [0], [0]])
    capacity = np.array([[40], [40], [40]])
    mu1 = 1  # Rate of job processing at buffer 1
    mu2 = 2  # Rate of job processing at buffer 2
    mu3 = 3  # Rate of job processing at buffer 3
    # Rows: buffer, columns: influence of activity.
    # Actions mean scheduling processing in one buffer.
    # There is no routing in this case, so this is a diagonal matrix.
    buffer_processing_matrix = np.array([[-mu1, 0, 0],
                                         [0, -mu2, 0],
                                         [0, 0, -mu3]])
    # Each row corresponds with a time-step. The resource can only work in one buffer at a time.
    actions = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1]])
    # When the resource works on a buffer, it compensate the arrivals, so the number of jobs remain
    # constant. Otherwise, the arrivals increases the number of jobs at the demand rate.
    jobs = np.array([[1, 2, 0],
                     [1, 4, 3],
                     [1, 6, 6],
                     [2, 6, 9],
                     [3, 8, 9],
                     [4, 8, 12],
                     [4, 10, 15],
                     [5, 12, 15],
                     [6, 12, 18],
                     [6, 14, 21],
                     [7, 16, 21]])
    # Expected cost Computed as dot product of cost_per_buffer and number of jobs at each iteration.
    cost = [5.5, 19.8, 34.1, 45.1, 50.6, 61.6, 75.9, 81.4, 92.4, 106.7, 112.2]

    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 3)), job_generator, s0)

    for i in range(11):
        s, r, d, t = env.step(actions[i].reshape((3, 1)))
        assert np.all(s == jobs[i].reshape([3, 1]))
        np.testing.assert_approx_equal(r, -cost[i])


def test_below_capacity_single_buffer():
    """One resource with one buffer. Check number of jobs is always equal or less than maximum
    capacity."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[3]])
    initial_state = np.array([[0]])
    capacity = np.array([[20]])

    job_generator = drjg(demand_rate, - np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    for i in range(10):
        s, r, d, t = env.step(action)
        assert 0 <= s <= capacity
    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_below_zero_capacity_single_buffer():
    """One resource with one buffer. Check number of jobs is always equal or less than maximum
    capacity, for the corner case of having zero maximum capacity."""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[3]])
    initial_state = np.array([[0]])
    capacity = np.array([[0]])

    job_generator = drjg(demand_rate, - np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_exceed_capacity_single_buffer():
    """One resource with one buffer. Check number of jobs is always equal or less than maximum
    capacity, for the corner case when initial_state > capacity"""
    cost_per_buffer = np.array([[3.5]])
    demand_rate = np.array([[3]])
    initial_state = np.array([[10]])
    capacity = np.array([[5]])

    job_generator = drjg(demand_rate, - np.ones((1, 1)), sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 1)), job_generator, s0)
    action = np.ones((1, 1))

    with pytest.raises(AssertionError):
        _ = env.step(action)


def test_exceed_capacity_multiple_buffer():
    """One resource with three buffers. Check number of jobs is always equal or less than maximum
    capacity."""
    cost_per_buffer = np.array([[1.1], [2.2], [3.3]])
    d1 = 1  # Rate of job arrival at buffer 1
    d2 = 2  # Rate of job arrival at buffer 2
    d3 = 3  # Rate of job arrival at buffer 3
    demand_rate = np.array([[d1], [d2], [d3]])
    initial_state = np.array([[0], [0], [0]])
    capacity = np.array([[4], [10], [13]])
    mu1 = 1  # Rate of job processing at buffer 1
    mu2 = 2  # Rate of job processing at buffer 2
    mu3 = 3  # Rate of job processing at buffer 3
    # Rows: buffer, columns: influence of activity.
    # Actions mean scheduling processing in one buffer.
    # There is no routing, so this is a diagonal matrix.
    buffer_processing_matrix = np.array([[-mu1, 0, 0],
                                         [0, -mu2, 0],
                                         [0, 0, -mu3]])
    # Each row corresponds with a time-step. The resource can only work in one buffer at a time.
    actions = np.array([[0, 0, 1],
                        [1, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 0]])
    # When we reach the maximum capacity, the amount of jobs remain constant, since processing rate
    # is equal to demand rate.
    jobs = np.array([[1, 2, 0],
                     [1, 4, 3],
                     [1, 6, 6],
                     [2, 6, 9],
                     [3, 8, 9],
                     [4, 8, 12],
                     [4, 10, 13]])
    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.ones((1, 3)), job_generator, s0)

    for i in range(6):
        s, r, d, t = env.step(actions[i].reshape((3, 1)))
        assert np.all(s == jobs[i].reshape([3, 1]))
    with pytest.raises(AssertionError):
        _ = env.step(actions[6].reshape((3, 1)))


def test_routing_two_serially_connected_resources_one_buffer_each():
    """Two resources with 1 buffer each. There are 2 possible actions. At every iteration, each
    resource chooses whether to work on its buffer or idle. If it works, jobs are processed at some
    rate. Jobs processed from resource 1 go to the buffer at resource 2. Events are generated
    deterministically at the rate given by the mean rate."""
    cost_per_buffer = np.array([[2], [4]])
    d1 = 1  # Rate of job arrival at buffer 1
    d2 = 0  # Rate of job arrival at buffer 2
    demand_rate = np.array([[d1], [d2]])
    initial_state = np.array([[0], [0]])
    capacity = np.array([[10], [10]])
    mu1 = 1  # Rate of job processing at buffer 1
    mu2 = 1  # Rate of job processing at buffer 2
    # Jobs processed at buffer 1 are routed to buffer 2.
    # Rows: buffer, columns: influence of activity.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0],
                                         [mu1, -mu2]])
    # activity 1 (column 0), processed job at buffer 1 (row 0) will be routed to buffer 2 (row 1).
    # Each row corresponds with a time-step. The resource can only work in one buffer at a time.
    actions = np.array([[0, 0],
                        [1, 0],
                        [1, 1],
                        [0, 1],
                        [1, 1]])
    # Expected number of jobs
    jobs = np.array([[1, 0],  # None buffer work. A new job gets to buffer 1 (b1)
                     [1, 1],  # Process b1 so job goes to buffer 2 (b2, then new job arrives at b1.
                     [1, 1],  # Process b1 and b2, since a new job arrives at b1, and a job is
                     # routed to b2, both buffers remain the same
                     [2, 0],  # Process b2 but not b1, so b1 accumulates one job, and b2 gets empty
                     [2, 0]])  # Process b1 and b2, so b1 has same number of jobs at the end, while
    # b2 aims to process the empty buffer and then gets one job routed from b1.
    # Expected cost
    cost = [2, 6, 6, 4, 4]

    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    assert job_generator.routes == {(1, 0): (0, 0)}
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)

    env = ControlledRandomWalk(cost_per_buffer, capacity, np.eye(2), job_generator, s0)

    for i in range(5):
        s, r, d, t = env.step(actions[i].reshape((2, 1)))
        assert np.all(s == jobs[i].reshape([2, 1]))
        np.testing.assert_approx_equal(r, -cost[i])


def env_job_conservation(job_conservation_flag):
    demand_rate = np.zeros((2, 1))
    cost_per_buffer = np.array([[2], [4]])
    initial_state = np.array([[0], [0]])
    capacity = np.array([[10], [10]])
    buffer_processing_matrix = np.array([[-1, 0],
                                         [1, -1]])
    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)
    return initial_state, ControlledRandomWalk(cost_per_buffer, capacity, np.eye(2),
                                               job_generator, s0, job_conservation_flag)


def test_job_conservation_flag_true():
    initial_state, env = env_job_conservation(True)
    s, r, d, t = env.step(np.array([1, 1])[:, None])
    assert np.all(s == initial_state)


def test_job_conservation_flag_false():
    initial_state, env = env_job_conservation(False)
    with pytest.raises(AssertionError):
        _ = env.step(np.array([1, 1])[:, None])


def test_job_conservation_multiple_routes_leave_same_buffer():
    demand_rate = np.zeros((3, 1))
    cost_per_buffer = np.ones((3, 1))
    initial_state = np.array([[3], [3], [1]])
    capacity = np.ones((3, 1)) * np.inf
    buffer_processing_matrix = np.array([[-2, -5, 0, 0],
                                         [2, 0, -10, 0],
                                         [0, 5, 0, -10]])
    constituency_mat = np.eye(4)
    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    job_conservation_flag = True
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_mat, job_generator, s0,
                               job_conservation_flag)
    s, r, d, t = env.step(np.array([1, 1, 1, 1])[:, None])
    # Depending on the order of activities we can get different result, since they are given by a
    # dictionary
    # In a event driven simulator, the order would be FIFO
    assert (s == np.array([[0], [2], [1]])).all() or (s == np.array([[0], [0], [3]])).all()


def test_routing_three_buffer_tree_topology():
    """
    Three resources with 1 buffer each, one parent with two children. There are 4 possible actions:
    children resources choose to idle or work on their respective buffer, while parent choose
    whether work and route to one child or work and route to the other. Events are generated
    deterministically at the rate given by the mean rate.
    """
    cost_per_buffer = np.array([[2], [4], [4]])
    d1 = 1  # Rate of job arrival at buffer 1
    d2 = 0  # Rate of job arrival at buffer 2
    d3 = 0  # Rate of job arrival at buffer 3
    demand_rate = np.array([[d1], [d2], [d3]])
    initial_state = np.array([[0], [0], [0]])
    capacity = np.array([[10], [10], [10]])
    mu12 = 1  # Rate of processing jobs at buffer 1 and routing to buffer 2
    mu13 = 1  # Rate of processing jobs at buffer 1 and routing to buffer 3
    mu2 = 1  # Rate of processing jobs at buffer 2
    mu3 = 1  # Rate of processing jobs at buffer 3
    # Jobs processed at buffer 1 are routed to either buffer 2 or 3.
    buffer_processing_matrix = np.array([[-mu12, -mu13, 0, 0],
                                         [mu12, 0, -mu2, 0],
                                         [0, mu13, 0, -mu3]])
    # Each row corresponds with a time-step. The resource can only work in one buffer at a time.
    actions = np.array([[0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 1, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 1],
                        [1, 0, 0, 1]])
    # Expected jobs
    jobs = np.array([[0, 0, 1],
                     [0, 0, 2],
                     [0, 1, 2],
                     [0, 1, 3],
                     [0, 1, 4],
                     [1, 0, 4],
                     [2, 0, 3],
                     [2, 0, 3],
                     [2, 1, 3],
                     [2, 1, 3],
                     [2, 2, 2]])

    job_generator = drjg(demand_rate, buffer_processing_matrix, sim_time_interval=1)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1)}
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state)
    constituency_matrix = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator, s0)

    for i in range(actions.shape[0]):
        s, r, d, t = env.step(actions[i].reshape((4, 1)))
        assert np.all(s == jobs[i].reshape([3, 1]))


def test_assert_surplus_buffers_consistent_with_job_generator_one_surplus_buffer():
    ind_surplus_buffers = [1]
    mud = 1e2
    buffer_processing_matrix = np.array([[-10, 100, 0],
                                         [10, 0, -mud],
                                         [0, 0, -mud]])
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(np.array([[0], [0], [9]]),
                                                                    buffer_processing_matrix)
    assert ControlledRandomWalk.is_surplus_buffers_consistent_with_job_generator(
        ind_surplus_buffers, job_generator.demand_nodes)


def test_assert_surplus_buffers_consistent_with_job_generator_multiple_surplus_buffer():
    ind_surplus_buffers = [4, 1]
    mud = 1e2
    buffer_processing_matrix = np.array([[-10, 100, 0, -10, 0],
                                         [10, 0, -mud, 0, 0],
                                         [0, 0, -mud, 0, 0],
                                         [0, 0, 0, 0, -mud],
                                         [0, 0, 0, 10, -mud]])
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(
        np.array([[0], [0], [0], [0], [9]]), buffer_processing_matrix)
    assert ControlledRandomWalk.is_surplus_buffers_consistent_with_job_generator(
        ind_surplus_buffers, job_generator.demand_nodes)


def test_assert_surplus_buffers_consistent_with_job_generator_wrong_one_surplus_buffer():
    ind_surplus_buffers = [2]
    mud = 1e2
    buffer_processing_matrix = np.array([[-10, 100, 0],
                                         [10, 0, -mud],
                                         [0, 0, -mud]])
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(np.array([[0], [0], [9]]),
                                                                    buffer_processing_matrix)
    assert not ControlledRandomWalk.is_surplus_buffers_consistent_with_job_generator(
        ind_surplus_buffers, job_generator.demand_nodes)


def test_assert_surplus_buffers_consistent_with_job_generator_single_station_demand_model():
    examples.single_station_demand_model()


def create_pull_model(ind_surplus_buffers):
    model_type = 'pull'
    mud = 1e2
    buffer_processing_matrix = np.array([[-10, 0, 100],
                                         [10, -mud, 0],
                                         [0, -mud, 0]])
    demand_rate = np.array([0, 0, 9])[:, None]
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix)
    return ControlledRandomWalk(np.array([1, 0.5, 10])[:, None],
                                np.ones((3, 1)) * np.inf,
                                np.eye(3),
                                job_generator,
                                DeterministicCRWStateInitialiser(demand_rate),
                                model_type=model_type,
                                ind_surplus_buffers=ind_surplus_buffers)


@pytest.mark.parametrize("ind_surplus_buffers", [1, None])
def test_assert_surplus_buffers_consistent_with_job_generator_invalid(ind_surplus_buffers):
    with pytest.raises(AssertionError):
        _ = create_pull_model(ind_surplus_buffers)


def test_ensure_jobs_conservation_with_enough_jobs():
    num_buffers = 2
    num_activities = 2
    buffer_processing_matrix = np.array([[-2, 0],
                                         [2, -3]])
    s0 = stinit.DeterministicCRWStateInitialiser(np.zeros((num_buffers, 1)))
    job_generator = drjg(np.ones((num_buffers, 1)), buffer_processing_matrix, sim_time_interval=1)
    env = ControlledRandomWalk(np.ones((num_buffers, 1)), np.ones((num_buffers, 1)),
                               np.zeros((num_buffers, num_activities)), job_generator, s0)

    state = 5 * np.ones((num_buffers, 1))
    routing_jobs_matrix = env.ensure_jobs_conservation(buffer_processing_matrix, state)
    assert np.all(routing_jobs_matrix == buffer_processing_matrix)


def test_ensure_jobs_conservation_with_not_enough_jobs():
    num_buffers = 2
    num_activities = 2
    buffer_processing_matrix = np.array([[-2, 0],
                                         [2, -3]])
    s0 = stinit.DeterministicCRWStateInitialiser(np.zeros((num_buffers, 1)))
    job_generator = drjg(np.ones((num_buffers, 1)), buffer_processing_matrix, sim_time_interval=1)
    env = ControlledRandomWalk(np.ones((num_buffers, 1)), np.ones((num_buffers, 1)),
                               np.zeros((num_buffers, num_activities)), job_generator, s0)

    state = np.ones((num_buffers, 1))
    routing_jobs_matrix = env.ensure_jobs_conservation(buffer_processing_matrix, state)
    assert np.all(routing_jobs_matrix == np.array([[-1, 0], [1, -1]]))


def test_ensure_jobs_conservation_with_zero_jobs():
    num_buffers = 2
    num_activities = 2
    buffer_processing_matrix = np.array([[-2, 0],
                                         [2, -3]])
    s0 = stinit.DeterministicCRWStateInitialiser(np.zeros((num_buffers, 1)))
    job_generator = drjg(np.ones((num_buffers, 1)), buffer_processing_matrix, sim_time_interval=1)
    env = ControlledRandomWalk(np.ones((num_buffers, 1)), np.ones((num_buffers, 1)),
                               np.zeros((num_buffers, num_activities)), job_generator, s0)

    state = np.zeros((num_buffers, 1))
    routing_jobs_matrix = env.ensure_jobs_conservation(buffer_processing_matrix, state)
    assert np.all(routing_jobs_matrix == np.zeros((num_buffers, num_activities)))


def test_ensure_jobs_conservation_with_multiple_demand_and_no_supply():
    num_buffers = 5
    num_activities = 5
    mu1 = 1
    mu2 = 2
    mu3 = 3
    mud1 = 4
    mud2 = 5
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0],
                                         [mu1, -mu2, mu3, 0, -mud2],
                                         [0, 0, 0, 0, -mud2],
                                         [0, mu2, -mu3, -mud1, 0],
                                         [0, 0, 0, -mud1, 0]])
    ind_surplus_buffers = [1, 3]
    s0 = stinit.DeterministicCRWStateInitialiser(np.zeros((num_buffers, 1)))
    job_generator = drjg(np.ones((num_buffers, 1)), buffer_processing_matrix, sim_time_interval=1)
    env = ControlledRandomWalk(
        np.ones((num_buffers, 1)),
        np.ones((num_buffers, 1)),
        np.zeros((num_buffers, num_activities)),
        job_generator,
        s0,
        model_type='pull',
        ind_surplus_buffers=ind_surplus_buffers
    )
    state = np.zeros((num_buffers, 1))
    routing_jobs_matrix = env.ensure_jobs_conservation(buffer_processing_matrix, state)
    assert np.all(routing_jobs_matrix == np.zeros((num_buffers, num_activities)))


def test_controlled_random_walk_reset():
    """Check that the CRW reset do reset its state and its job generator seed"""
    cost_per_buffer = np.array([[3.5]])
    capacity = np.ones((1, 1)) * np.inf
    constituency_matrix = np.ones((1, 1))
    demand_rate = np.array([[1000]])
    buffer_processing_matrix = np.array([[5]])
    initial_state = np.array([[20]])
    seed = 42

    job_generator = prjg(sim_time_interval=1, demand_rate=demand_rate,
                         buffer_processing_matrix=buffer_processing_matrix, job_gen_seed=seed)
    initial_random_state = job_generator.np_random.get_state()[1]
    s0 = stinit.DeterministicCRWStateInitialiser(initial_state=initial_state)

    env = ControlledRandomWalk(cost_per_buffer=cost_per_buffer, capacity=capacity,
                               constituency_matrix=constituency_matrix, job_generator=job_generator,
                               state_initialiser=s0)

    next_state, _, _, _ = env.step(action=np.ones((1, 1)))
    next_random_state = job_generator.np_random.get_state()[1]
    assert np.any(next_state != initial_state)
    assert np.any(next_random_state != initial_random_state)

    env.reset_with_random_state()
    new_initial_state = env.state
    new_initial_random_state = job_generator.np_random.get_state()[1]
    assert np.all(new_initial_state == initial_state)
    assert np.all(new_initial_random_state == initial_random_state)


def test_job_generator_reset():
    """Check that the Job Generator reset do reset its seed"""
    demand_rate = np.array([[1000]])
    buffer_processing_matrix = np.array([[5]])
    seed = 42

    job_generator = prjg(sim_time_interval=1, demand_rate=demand_rate,
                         buffer_processing_matrix=buffer_processing_matrix, job_gen_seed=seed)
    initial_random_state = job_generator.np_random.get_state()[1]

    _ = job_generator.get_arrival_jobs()
    next_random_state = job_generator.np_random.get_state()[1]
    assert np.any(next_random_state != initial_random_state)

    job_generator.reset_seed()
    new_initial_random_state = job_generator.np_random.get_state()[1]
    assert np.all(new_initial_random_state == initial_random_state)


def test_job_generator_prjg_fixed_seed():
    """Check that the methods from two different instance of a PoissonDiscreteReviewJobGenerator
    return the same results given the same seed."""
    demand_rate = np.array([[1000]])
    supply_rate = 1000.
    buffer_processing_matrix = np.array([[5]])
    seed = 42
    np.random.seed(seed)

    job_generator_1 = prjg(sim_time_interval=1, demand_rate=demand_rate,
                           buffer_processing_matrix=buffer_processing_matrix,
                           job_gen_seed=seed)

    job_generator_2 = prjg(sim_time_interval=1, demand_rate=demand_rate,
                           buffer_processing_matrix=buffer_processing_matrix,
                           job_gen_seed=seed)

    for _ in np.arange(1000):
        arrival_jobs_1 = job_generator_1.get_arrival_jobs()
        arrival_jobs_2 = job_generator_2.get_arrival_jobs()
        assert np.all(arrival_jobs_1 == arrival_jobs_2)

        drained_jobs_matrix_1 = job_generator_1.get_instantaneous_drained_jobs_matrix()
        drained_jobs_matrix_2 = job_generator_2.get_instantaneous_drained_jobs_matrix()
        assert np.all(drained_jobs_matrix_1 == drained_jobs_matrix_2)

        supplied_jobs_1 = job_generator_1.get_supplied_jobs(rate=supply_rate)
        supplied_jobs_2 = job_generator_2.get_supplied_jobs(rate=supply_rate)
        assert np.all(supplied_jobs_1 == supplied_jobs_2)


def test_create_example_env():
    envs = [examples.dai_wang_model,
            examples.input_queued_switch_3x3_model,
            examples.klimov_model,
            examples.ksrs_network_model,
            examples.processor_sharing_model,
            examples.routing_with_negative_workload,
            examples.simple_link_constrained_model,
            examples.simple_link_constrained_with_route_scheduling_model,
            examples.loop_2_queues,
            examples.simple_reentrant_line_model,
            examples.simple_routing_model,
            examples.single_server_queue,
            examples.three_station_network_model,
            # Pull models
            examples.double_reentrant_line_model,
            examples.double_reentrant_line_only_shared_resources_model,
            examples.double_reentrant_line_with_demand_model,
            examples.double_reentrant_line_with_demand_only_shared_resources_model,
            examples.complex_demand_driven_model,
            examples.multiple_demand_model,
            examples.simple_reentrant_line_with_demand_model,
            examples.single_station_demand_model,
            examples.willems_example_2]

    for e in envs:
        _ = e()


def test_create_distribution_with_rebalancing_example_env():
    examples_distribution_with_rebalancing.one_warehouse()

    examples_distribution_with_rebalancing.two_warehouses(r_to_w_rebalance=False,
                                                          w_to_w_rebalance=False)

    examples_distribution_with_rebalancing.two_warehouses(r_to_w_rebalance=True,
                                                          w_to_w_rebalance=False)

    examples_distribution_with_rebalancing.two_warehouses(r_to_w_rebalance=False,
                                                          w_to_w_rebalance=True)

    examples_distribution_with_rebalancing.two_warehouses(r_to_w_rebalance=True,
                                                          w_to_w_rebalance=True)

    examples_distribution_with_rebalancing.two_warehouses_simplified(r_to_w_rebalance=False,
                                                                     w_to_w_rebalance=False)

    examples_distribution_with_rebalancing.two_warehouses_simplified(r_to_w_rebalance=True,
                                                                     w_to_w_rebalance=False)

    examples_distribution_with_rebalancing.two_warehouses_simplified(r_to_w_rebalance=False,
                                                                     w_to_w_rebalance=True)

    examples_distribution_with_rebalancing.two_warehouses_simplified(r_to_w_rebalance=True,
                                                                     w_to_w_rebalance=True)

    examples_distribution_with_rebalancing.three_warehouses_simplified(r_to_w_rebalance=True)

    examples_distribution_with_rebalancing.three_warehouses_simplified(r_to_w_rebalance=False)

    examples_distribution_with_rebalancing.three_warehouses()

    examples_distribution_with_rebalancing.three_warehouses_two_manufacturers_per_area()
