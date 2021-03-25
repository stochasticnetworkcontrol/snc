# Set of examples of systems that can be modelled as controlled random walks, from CTCN book.

"""
*Explanation of boundary constraints data structure

list_boundary_constraint_matrices is a list (of np.ndarray type) of length equal to the
number of resources.
Each element of the list corresponds to a single resource, has a shape
(number of constraints, number of buffers)
and for each constraint row has a value of 1 at buffer indices which need to be non-empty
for the resource to be able to work and value 0 otherwise.

Note that constraints are per resource, not per activity.

For demand pull models, fictitious resource draining demand buffer has no constraints and therefore
its constraint matrix is all zeros.


*Explanation on the route representation in job_generator objects

job_generator.routes has a given structure by convention:
{(target_buffer_index, activity_index):(source_buffer_index, activity_index)}

activity_index is the same in key-value pairs
"""
import numpy as np
from typing import Optional, Dict, Tuple
from src.snc.environments import ControlledRandomWalk
from src.snc.environments \
    import DeterministicDiscreteReviewJobGenerator
from src.snc.environments import JobGeneratorInterface
from src.snc.environments \
    import ScaledBernoulliServicesAndArrivalsGenerator
from src.snc.environments \
    import ScaledBernoulliServicesGeometricXBernoulliArrivalsGenerator
from src.snc.environments \
    import ScaledBernoulliServicesPoissonArrivalsGenerator
from src.snc.environments.state_initialiser import DeterministicCRWStateInitialiser

# @TODO:
#  Refactor examples as list of parameters so we have more flexibility to call them with different
#  state initialisers and job generators.

# This file contains examples for
# - single server queue
# - simple routing model
# - Klimov model
# - processor sharing model
# - simple re-entrant line model
# - double re-entrant line model
# - double re-entrant line only shared resources model
# - KSRS network model
# - simple link constrained
# - simple link constrained with route scheduling
# - input queued switch 3X3 model
# - three station network model
# - single station demand model
# - multiple demand model
# - simple re-entrant line with demand model
# - double re-entrant line with demand model
# - double re-entrant line with demand only shared resources model
# - complex demand driven model = routing, scheduling, and exogenous demand, hot-lots
# - Willems example 2 (a resource can perform either a single activity or multiple activities
#   Note: items arriving in a buffer from different activities are not combined in a single item.

# TODO: check if the boundary constraints in the examples 'with demand' are correct


def single_server_queue(cost_per_buffer=np.ones((1, 1)), initial_state=np.zeros((1, 1)),
                        capacity=np.ones((1, 1)) * np.inf, demand_rate_val=0.7,
                        job_conservation_flag=True, job_gen_seed: Optional[int] = None,
                        max_episode_length: Optional[int] = None, deterministic: bool = False) \
        -> ControlledRandomWalk:
    """Simplest model: One buffer and one resource."""
    # Problem data.
    demand_rate = np.array([demand_rate_val])[:, None]
    buffer_processing_matrix = - np.ones((1, 1))
    constituency_matrix = np.ones((1, 1))
    list_boundary_constraint_matrices = [constituency_matrix]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed
        )
    else:
        job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                        buffer_processing_matrix,
                                                                        job_gen_seed=job_gen_seed)

    assert job_generator.routes == {}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def simple_routing_model(alpha_r=0.2,
                         mu1=0.13, mu2=0.07, mu_r=0.2,
                         cost_per_buffer=np.ones((3, 1)),
                         initial_state=(1, 1, 1),
                         capacity=np.ones((3, 1)) * np.inf,
                         job_conservation_flag=True,
                         job_gen_seed: Optional[int] = None,
                         max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Station with one buffer and two routes, each one these resources representing a route to
    different buffers, each one in a different station."""
    # Problem data.
    demand_rate = np.array([0, 0, alpha_r])[:, None]
    # For buffer processing matrix rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, mu_r, 0],
                                         [0, -mu2, 0, mu_r],
                                         [0, 0, -mu_r, -mu_r]])
    constituency_matrix = np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 1]])
    # Resource 1 needs buffer 1 to have customers.
    list_boundary_constraint_matrices = [np.array([[1, 0, 0]]),
                                         np.array([[0, 1, 0]]),
                                         np.array([[0, 0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(0, 2): (2, 2), (1, 3): (2, 3)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def klimov_model(alpha1=0.2, alpha2=.3, alpha3=.4, alpha4=.5,
                 mu1=1.1, mu2=2.2, mu3=3.3, mu4=4.4,
                 cost_per_buffer=np.ones((4, 1)),
                 initial_state=(0, 0, 0, 0),
                 capacity=np.ones((4, 1)) * np.inf,
                 job_conservation_flag=True,
                 job_gen_seed: Optional[int] = None,
                 max_episode_length: Optional[int] = None,
                 deterministic: bool = False) -> ControlledRandomWalk:
    """Example 4.2.1, Figure 2.3 from CTCN book."""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [0, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, 0, -mu4]])
    constituency_matrix = np.array([[1, 1, 1, 1]])
    demand_rate = np.array([alpha1, alpha2, alpha3, alpha4])[:, None]
    list_boundary_constraint_matrices = [constituency_matrix]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                        buffer_processing_matrix,
                                                                        job_gen_seed=job_gen_seed)

    assert job_generator.routes == {}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def converging_push_model(alpha=1, mu1=1.02, mu2=1.02, mu3=2.08,
                          cost_per_buffer=np.array([1, 1.5, 2])[:, None],
                          initial_state=np.array([1, 1, 100])[:, None],
                          capacity=np.ones((3, 1)) * np.inf,
                          job_conservation_flag=True,
                          job_gen_seed: Optional[int] = None,
                          max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """
    Very simple network consisting of three buffers and three single activity resources.
    Together with specially chosen default parameter settings its purpose is to demonstrate
    guaranteed superior performance of hedgehog to myopic and MaxWeight policies.
    Or rather to show how these benchmarks diverge from optimum behaviour.
    """
    buffer_processing_matrix = np.array([[-mu1, 0, 0],
                                         [0, -mu2, 0],
                                         [mu1, mu2, -mu3]])
    constituency_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
    demand_rate = np.array([alpha, alpha, 0])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0]]),
                                         np.array([[0, 1, 0]]),
                                         np.array([[0, 0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(2, 0): (0, 0),
                                    (2, 1): (1, 1)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def simple_reentrant_line_model(alpha1=9, mu1=22, mu2=10, mu3=22,
                                cost_per_buffer=np.ones((3, 1)),
                                initial_state=np.array([[0], [0], [0]]),
                                capacity=np.ones((3, 1)) * np.inf,
                                job_conservation_flag=True, job_gen_seed: Optional[int] = None,
                                max_episode_length: Optional[int] = None,
                                deterministic: bool = False) -> ControlledRandomWalk:
    """Example 4.2.3, Figure 2.9  from CTCN book."""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0],
                                         [mu1, -mu2, 0],
                                         [0, mu2, -mu3]])
    constituency_matrix = np.array([[1, 0, 1],
                                    [0, 1, 0]])
    demand_rate = np.array([alpha1, 0, 0])[:, None]
    list_boundary_constraint_matrices = [constituency_matrix[0, :][None, :],
                                         constituency_matrix[1, :][None, :]]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesAndArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (1, 1)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def extended_reentrant_line_model(alpha1=0.33, mu1=0.69, mu2=0.35, mu3=0.68, mu4=0.35,
                                  cost_per_buffer=np.array([1.5, 1, 2, 3, 2, 5])[:, None],
                                  initial_state=np.array([1000, 0, 500, 300, 1000, 0])[:, None],
                                  capacity=np.ones((6, 1)) * np.inf,
                                  job_conservation_flag=True, job_gen_seed: Optional[int] = None,
                                  max_episode_length: Optional[int] = None,
                                  deterministic: bool = False) -> ControlledRandomWalk:
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0, 0],
                                         [mu1, -mu2, 0, 0, 0, 0],
                                         [0, mu2, -mu1, 0, 0, 0],
                                         [0, 0, mu1, -mu3, 0, 0],
                                         [0, 0, 0, 0, -mu3, 0],
                                         [0, 0, 0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 1, 0, 0, 0],
                                    [0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    demand_rate = np.array([alpha1, 0, 0, 0, alpha1, 0])[:, None]
    list_boundary_constraint_matrices = [constituency_matrix[0, :][None, :],
                                         constituency_matrix[1, :][None, :],
                                         constituency_matrix[2, :][None, :],
                                         constituency_matrix[3, :][None, :],
                                         ]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesAndArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (1, 1), (3, 2): (2, 2),
                                    (5,4): (4,4)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def decoupled_simple_reentrant_line_models(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68, mu4=0.99,
                                           cost_per_buffer=np.ones((7, 1)),
                                           initial_state=np.zeros((7, 1)),
                                           capacity=np.ones((7, 1)) * np.inf,
                                           job_conservation_flag=True,
                                           job_gen_seed: Optional[int] = None,
                                           max_episode_length: Optional[int] = None,
                                           deterministic: bool = False) -> ControlledRandomWalk:
    """
    Example designed to study whether the idling decision takes into account
    both resources or just one.
    """
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.zeros((7,7))
    buffer_processing_matrix[0:3 ,0:3] = np.array([[-mu1, 0, 0],
                                                   [mu1, -mu2, 0],
                                                   [0, mu2, -mu3]])
    buffer_processing_matrix[3:6, 3:6] = np.array([[-mu1, 0, 0],
                                                   [mu1, -mu2, 0],
                                                   [0, mu2, -mu3]])
    buffer_processing_matrix[6, [2, 5]] = mu3
    buffer_processing_matrix[6, 6] = -mu4
    constituency_matrix = np.zeros((5,7))

    constituency_matrix[0:2, 0:3] = np.array([[1, 0, 1],
                                              [0, 1, 0]])
    constituency_matrix[2:4, 3:6] = np.array([[1, 0, 1],
                                              [0, 1, 0]])
    constituency_matrix[4, 6] = 1
    demand_rate = np.array([alpha1, 0, 0, alpha1, 0, 0, 0])[:, None]
    list_boundary_constraint_matrices = []
    for i,_ in enumerate(constituency_matrix):
        list_boundary_constraint_matrices.append(constituency_matrix[i, :][None, :].copy())
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesAndArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (1, 1), (6, 2): (2, 2),
                                    (4, 3): (3, 3), (5, 4): (4, 4), (6, 5): (5, 5)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def single_reentrant_resource(alpha1=0.33, mu1=0.68, mu2=0.68,
                              cost_per_buffer=np.ones((2, 1)),
                              initial_state=np.array([[0], [0]]),
                              capacity=np.ones((2, 1)) * np.inf,
                              job_conservation_flag=True, job_gen_seed: Optional[int] = None,
                              max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """A network with a single resource responsible for two activities in serial configuration"""
    buffer_processing_matrix = np.array([[-mu1, 0],
                                         [mu1, -mu2]])
    constituency_matrix = np.array([[1, 1]])
    demand_rate = np.array([alpha1, 0])[:, None]
    list_boundary_constraint_matrices = [constituency_matrix]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    job_generator = ScaledBernoulliServicesAndArrivalsGenerator(demand_rate,
                                                                buffer_processing_matrix,
                                                                job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def simple_reentrant_line_model_variance(
        alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68,
        demand_variance=0.33,
        cost_per_buffer=np.ones((3, 1)),
        initial_state=np.array([[0], [0], [0]]),
        job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    """
    Variant of the simple reentrant line model, where we replace the Bernoulli job generator with a
    geometric times Bernoulli job generator that allows us to control the variance.
    """
    # Build standard model.
    env = simple_reentrant_line_model(alpha1, mu1, mu2, mu3, cost_per_buffer, initial_state)
    # Replace job generator.
    demand_variance_vec = np.array([[demand_variance], [0], [0]])
    demand_rate = env.job_generator.demand_rate
    buffer_processing_matrix = env.job_generator.buffer_processing_matrix
    job_generator = ScaledBernoulliServicesGeometricXBernoulliArrivalsGenerator(
        demand_rate, demand_variance_vec, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    env.job_generator = job_generator
    return env


def double_reentrant_line_model(alpha=0.25, mu1=1, mu2=0.75, mu3=0.5, mu4=0.75, mu5=1,
                                cost_per_buffer=np.array([1, 1, 1, 1, 1])[:, None],
                                initial_state=np.array([1, 1, 1, 1, 1])[:, None],
                                capacity=np.ones((5, 1)) * np.inf,
                                job_conservation_flag=True,
                                job_gen_seed: Optional[int] = None,
                                max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """
    Double re-entrant line. As the single re-entrant line but with an extra resource.
    """
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0],
                                         [mu1, -mu2, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0],
                                         [0, 0, mu3, -mu4, 0],
                                         [0, 0, 0, mu4, -mu5]])
    constituency_matrix = np.array([[1, 0, 0, 0, 1],
                                    [0, 1, 0, 1, 0],
                                    [0, 0, 1, 0, 0]])
    demand_rate = np.array([alpha, 0, 0, 0, 0])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 1]]),
                                         np.array([[0, 1, 0, 1, 0]]),
                                         np.array([[0, 0, 1, 0, 0]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (4, 3): (3, 3)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def double_reentrant_line_only_shared_resources_model(alpha=1, mu1=4, mu2=3, mu3=3, mu4=4,
                                                      cost_per_buffer=np.array([1, 1, 1, 1])
                                                      [:, None],
                                                      initial_state=np.array([1, 1, 1, 1])[:, None],
                                                      capacity=np.ones((4, 1)) * np.inf,
                                                      job_conservation_flag=True,
                                                      job_gen_seed: Optional[int] = None,
                                                      max_episode_length: Optional[int] = None,
                                                      deterministic: bool = False) \
        -> ControlledRandomWalk:
    """Double re-entrant line where all the resources have two activities.
    As in Figure 7.11 (printed book) but without demand"""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, mu2, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])
    demand_rate = np.array([alpha, 0, 0, 0])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 1]]),
                                         np.array([[0, 1, 1, 0]])]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                        buffer_processing_matrix,
                                                                        job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def processor_sharing_model(alpha_a=2, alpha_b=2,
                            mu_a=1, mu_b=4, mu_c=3,
                            cost_per_buffer=np.ones((2, 1)),
                            initial_state=(0, 0),
                            capacity=np.ones((2, 1)) * np.inf,
                            job_conservation_flag=True,
                            job_gen_seed: Optional[int] = None,
                            max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 4.2.2, Figure 2.6 from CTCN book."""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu_a, -mu_b, 0],
                                         [0, 0, -mu_c]])
    constituency_matrix = np.array([[1, 0, 0],
                                    [0, 1, 1]])
    demand_rate = np.array([alpha_a, alpha_b])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0]]), np.array([[1, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def ksrs_network_model(alpha1=2, alpha3=2,
                       mu1=10, mu2=3, mu3=10, mu4=3,
                       cost_per_buffer=np.ones((4, 1)),
                       initial_state=(0, 0, 0, 0),
                       capacity=np.ones((4, 1)) * np.inf,
                       job_conservation_flag=True,
                       job_gen_seed: Optional[int] = None,
                       list_boundary_constraint_matrices=None,
                       max_episode_length: Optional[int] = None,
                       deterministic: bool = False) -> ControlledRandomWalk:
    """Example 4.2.4, and Sec. 2.9 (Figure 2.12) from CTCN book."""
    demand_rate = np.array([alpha1, 0, alpha3, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0],
                                         [mu1, -mu2, 0, 0],
                                         [0, 0, -mu3, 0],
                                         [0, 0, mu3, -mu4]])
    constituency_matrix = np.array([[1, 0, 0, 1],
                                    [0, 1, 1, 0]])
    if list_boundary_constraint_matrices is None:
        list_boundary_constraint_matrices = [constituency_matrix[0, :][None, :],
                                             constituency_matrix[1, :][None, :]]
    model_type = 'push'

    # Construct environment.
    # Set up the job generator as stochastic so that we can use the time interval calculation.
    # If a deterministic generator is needed then up can update it.
    # Type is any to satisfy mypy and the change of type of the job generator while the parent
    # class JobGenerator does not have a sim_time_interval property,
    if deterministic:
        # Change the job generator for a deterministic one using the calculated sim_time_interval
        # from the stochastic job generator.
        job_generator: JobGeneratorInterface = DeterministicDiscreteReviewJobGenerator(
            demand_rate,
            buffer_processing_matrix,
            job_gen_seed=job_gen_seed)
    else:
        job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                        buffer_processing_matrix,
                                                                        job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0), (3, 2): (2, 2)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def simple_link_constrained_model(alpha1=4, mu12=2, mu13=10, mu25=1, mu32=5, mu34=2, mu35=2,
                                  mu45=10, mu5=10, cost_per_buffer=np.ones((5, 1)),
                                  initial_state=np.zeros((5, 1)), capacity=np.ones((5, 1)) * np.inf,
                                  job_conservation_flag=False, job_gen_seed: Optional[int] = None,
                                  max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 6.3.1. Simple link-constrained model from CTCN online (Example 6.3.5 from printed
    version). Corresponds with Figure 6.7."""
    demand_rate = np.array([alpha1, 0, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu12, -mu13, 0, 0, 0, 0, 0, 0],
                                         [mu12, 0, -mu25, mu32, 0, 0, 0, 0],
                                         [0, mu13, 0, -mu32, -mu34, -mu35, 0, 0],
                                         [0, 0, 0, 0, mu34, 0, -mu45, 0],
                                         [0, 0, mu25, 0, 0, mu35, mu45, -mu5]])
    # {(d, a): (s, a)} indicates that we are routing from buffer s to buffer d when taking action a.
    constituency_matrix = np.eye(8)
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1),
                                    (4, 2): (1, 2), (1, 3): (2, 3),
                                    (3, 4): (2, 4), (4, 5): (2, 5),
                                    (4, 6): (3, 6)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def simple_link_constrained_with_route_scheduling_model(alpha1=2.9, mu12=2, mu13=10, mu25=1,
                                                        mu32=5, mu34=2, mu35=2, mu45=10, mu5=100,
                                                        cost_per_buffer=np.ones((5, 1)),
                                                        initial_state=np.zeros((5, 1)),
                                                        capacity=np.ones((5, 1)) * np.inf,
                                                        job_conservation_flag=True,
                                                        job_gen_seed: Optional[int] = None,
                                                        max_episode_length: Optional[int] = None) \
        -> ControlledRandomWalk:
    """
    We follow Figure 6.7 from CTCN book, but we model the resources as being able to schedule one
    output route at a time, as opposed to Example 6.3.1 where each link is a resource.
    """
    demand_rate = np.array([alpha1, 0, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu12, -mu13, 0, 0, 0, 0, 0, 0],
                                         [mu12, 0, -mu25, mu32, 0, 0, 0, 0],
                                         [0, mu13, 0, -mu32, -mu34, -mu35, 0, 0],
                                         [0, 0, 0, 0, mu34, 0, -mu45, 0],
                                         [0, 0, mu25, 0, 0, mu35, mu45, -mu5]])
    # {(d, a): (s, a)} indicates that we are routing from buffer s to buffer d when taking action a.
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0), (2, 1): (0, 1),
                                    (4, 2): (1, 2), (1, 3): (2, 3),
                                    (3, 4): (2, 4), (4, 5): (2, 5),
                                    (4, 6): (3, 6)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def routing_with_negative_workload(alpha1=0.45, mu1=0.5, mu2=0.9, mu3=0.8,
                                   cost_per_buffer=np.ones((2, 1)),
                                   initial_state=np.zeros((2, 1)),
                                   capacity=np.ones((2, 1)) * np.inf,
                                   job_conservation_flag=True,
                                   job_gen_seed: Optional[int] = None):
    demand_rate = np.array([[alpha1], [0]])
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0], [mu1, 0, -mu3]])
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 0]]),
                                         np.array([[0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def routing_with_negative_workload_and_double_draining_action(
        alpha1=0.45, mu1=0.5, mu2=0.9, mu3=0.8,
        cost_per_buffer=np.ones((2, 1)),
        initial_state=np.zeros((2, 1)),
        capacity=np.ones((2, 1)) * np.inf,
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None):

    demand_rate = np.array([[alpha1], [0]])
    buffer_processing_matrix = np.array([[-mu1, -mu2, 0], [mu1, -mu2, -mu3]])
    constituency_matrix = np.array([[1, 1, 0], [0, 0, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 0], [0, 1]]),
                                         np.array([[0, 1]])]
    model_type = 'push'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def loop_2_queues(mu1=1, mu12=0.5, mu2=0.8, mu21=1,
                  cost_per_buffer=np.array([[1], [0.1]]),
                  initial_state=np.zeros((2, 1)),
                  demand_rate=np.array([[0.1], [0.95]]),
                  capacity=np.ones((2, 1)) * np.inf,
                  job_conservation_flag=True,
                  job_gen_seed: Optional[int] = None) -> ControlledRandomWalk:
    buffer_processing_matrix = np.array([[-mu1, -mu12, 0, mu21],
                                         [0, mu12, -mu2, -mu21]])
    constituency_matrix = np.array([[1, 1, 0, 0],
                                    [0, 0, 1, 1]])
    list_boundary_constraint_matrices = [np.array([[0, 0]]),
                                         np.array([[0, 1]])]
    model_type = 'push'

    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 1): (0, 1),  # mu12 (action 1) routes from buffer 0 to 1.
                                    (0, 3): (1, 3)}  # mu21 (action 3) routes from buffer 1 to 0.
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type)
    return env


def input_queued_switch_3x3_model(
        mu11=1, mu12=1, mu13=1, mu21=1, mu22=1, mu23=1, mu31=1, mu32=1, mu33=1,
        cost_per_buffer=np.ones((9, 1)),
        initial_state=np.zeros((9, 1)),
        capacity=np.ones((9, 1)) * np.inf,
        demand_rate=0.3 * np.ones((9, 1)),
        job_conservation_flag=True,
        job_gen_seed: Optional[int] = None,
        max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 6.5.2, Figure 6.10, Proposition 6.6.2, from CTCN book."""
    buffer_processing_matrix = - np.diag([mu11, mu12, mu13, mu21, mu22, mu23, mu31, mu32, mu33])
    constituency_matrix = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 1, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 1, 1],
                                    [1, 0, 0, 1, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                                    [0, 0, 1, 0, 0, 1, 0, 0, 1]])
    list_boundary_constraint_matrices = []
    for c in constituency_matrix:
        list_boundary_constraint_matrices.append(c[None, :])
    model_type = 'push'
    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def three_station_network_model(alpha1=0, alpha6=0,
                                mu1=1, mu2=1, mu3=1, mu4=1, mu5=1, mu6=1,
                                cost_per_buffer=np.ones((6, 1)),
                                initial_state=np.zeros((6, 1)),
                                capacity=np.ones((6, 1)) * np.inf,
                                job_conservation_flag=True,
                                job_gen_seed: Optional[int] = None,
                                max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 5.3.2 from CTCN online (Example 5.3.6 from printed version). Figure 5.2."""
    demand_rate = np.array([alpha1, 0, 0, 0, 0, alpha6])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0, 0],
                                         [0, -mu2, 0, mu4, 0, 0],
                                         [mu1, 0, -mu3, 0, 0, 0],
                                         [0, 0, 0, -mu4, 0, mu6],
                                         [0, 0, mu3, 0, -mu5, 0],
                                         [0, 0, 0, 0, 0, -mu6]])
    constituency_matrix = np.array([[1, 1, 0, 0, 0, 0],
                                    [0, 0, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 1, 1]])
    list_boundary_constraint_matrices = [np.array([[1, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 1, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 1, 1]])]
    model_type = 'push'
    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    # {(d, a): (s, a)} indicates that we are routing from buffer s to buffer d when taking action a.
    assert job_generator.routes == {(2, 0): (0, 0),
                                    (4, 2): (2, 2),
                                    (1, 3): (3, 3),
                                    (3, 5): (5, 5)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def dai_wang_model(alpha1=0.2, mu1=0.66, mu2=0.66, mu3=0.42, mu4=0.42, mu5=0.66,
                   demand_variance=0.2,
                   cost_per_buffer=np.array([[1], [2], [2], [2], [2]]),
                   initial_state=50 * np.ones((5, 1)),
                   capacity=np.ones((5, 1)) * np.inf,
                   job_conservation_flag=False,
                   job_gen_seed: Optional[int] = None,
                   max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 4.6.4 and Figure 4.15 from CTCN online ed."""
    demand_variance_vec = np.array([[demand_variance], [0], [0], [0], [0]])
    demand_rate = np.array([alpha1, 0, 0, 0, 0])[:, None]
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0],
                                         [mu1, -mu2, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0],
                                         [0, 0, mu3, -mu4, 0],
                                         [0, 0, 0, mu4, -mu5]])
    constituency_matrix = np.array([[1, 1, 0, 0, 1],
                                    [0, 0, 1, 1, 0]])
    list_boundary_constraint_matrices = [np.array([[1, 1, 0, 0, 1]]),
                                         np.array([[0, 0, 1, 1, 0]])]
    model_type = 'push'
    # Construct environment.
    job_generator = ScaledBernoulliServicesGeometricXBernoulliArrivalsGenerator(
        demand_rate, demand_variance_vec, buffer_processing_matrix, job_gen_seed=job_gen_seed)
    # {(d, a): (s, a)} indicates that we are routing from buffer s to buffer d when taking action a.
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (4, 3): (3, 3)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def single_station_demand_model(alpha_d=9,
                                mu=10, mus=1e2, mud=1e2,
                                cost_per_buffer=np.array([1, 0.5, 10])[:, None],
                                initial_state=np.array(([30, 450, 0])),
                                capacity=np.ones((3, 1)) * np.inf,
                                job_conservation_flag=False,
                                job_gen_seed: Optional[int] = None,
                                max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 7.1.1. Single-station demand-driven model from the CTCN book"""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu, 0, mus],
                                         [mu, -mud, 0],
                                         [0, -mud, 0]])
    constituency_matrix = np.eye(3)
    demand_rate = np.array([0, 0, alpha_d])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0]]),
                                         np.array([[0, 1, 0]]),
                                         np.array([[0, 0, 0]])]
    model_type = 'pull'
    ind_surplus_buffers = [1]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer,
                               capacity,
                               constituency_matrix,
                               job_generator,
                               state_initialiser,
                               job_conservation_flag,
                               list_boundary_constraint_matrices,
                               model_type,
                               ind_surplus_buffers=ind_surplus_buffers,
                               max_episode_length=max_episode_length)
    return env


def multiple_demand_model(d1=5, d2=5, mu1=12, mu2=9, mu3=12, mu4=9, mus=100, mud=100,
                          cost_per_buffer=np.array([1, 1, 1, 1, 5, 10, 5, 10])[:, None],
                          initial_state=np.array(([100, 25, 75, 100, 0, 200, 25, 0])),
                          capacity=np.ones((8, 1)) * np.inf,
                          job_conservation_flag=False,
                          job_gen_seed: Optional[int] = None,
                          max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Example 7.2.2. Multiple demands (Figure 7.7) in both the CTCN and the printed book"""
    buffer_processing_matrix = np.zeros((8, 8))
    buffer_processing_matrix[0, 0] = -mu1
    buffer_processing_matrix[1, 0] = mu1
    buffer_processing_matrix[1, 1] = -mu2
    buffer_processing_matrix[4, 1] = mu2
    buffer_processing_matrix[2, 2] = -mu3
    buffer_processing_matrix[3, 2] = mu3
    buffer_processing_matrix[3, 3] = -mu4
    buffer_processing_matrix[6, 3] = mu4
    buffer_processing_matrix[4, 4] = -mud
    buffer_processing_matrix[5, 4] = -mud
    buffer_processing_matrix[6, 5] = -mud
    buffer_processing_matrix[7, 5] = -mud
    buffer_processing_matrix[0, 6] = mus
    buffer_processing_matrix[2, 7] = mus
    constituency_matrix = np.array([[1, 0, 0, 1, 0, 0, 0, 0],
                                    [0, 1, 1, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 1]])
    demand_rate = np.array([0, 0, 0, 0, 0, d1, 0, d2])[:, None]
    ind_surplus_buffers = [4, 6]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 1, 1, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (4, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (6, 3): (3, 3)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers,
        max_episode_length=max_episode_length
    )
    return env


def simple_reentrant_line_with_demand_model(alpha_d=2,
                                            mu1=5, mu2=2.5, mu3=5, mus=1e3, mud=1e3,
                                            cost_per_buffer=np.ones((5, 1)),
                                            initial_state=np.array([10, 25, 55, 0, 100])[:, None],
                                            capacity=np.ones((5, 1)) * np.inf,
                                            job_conservation_flag=True,
                                            job_gen_seed: Optional[int] = None,
                                            max_episode_length: Optional[int] = None) \
        -> ControlledRandomWalk:
    """ Example 7.1.3. Simple re-entrant line with demand, Figure 7.5  from CTCN book (online ed).
    Example 7.1.4. from the printed book. """
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, mus],
                                         [mu1, -mu2, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0],
                                         [0, 0, mu3, -mud, 0],
                                         [0, 0, 0, -mud, 0]])
    constituency_matrix = np.array([[1, 0, 1, 0, 0],
                                    [0, 1, 0, 0, 0],
                                    [0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 1]])
    demand_rate = np.array([0, 0, 0, 0, alpha_d])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 1, 0, 0]]),
                                         np.array([[0, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 0]])]
    model_type = 'pull'
    ind_surplus_buffers = [3]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               ind_surplus_buffers=ind_surplus_buffers,
                               max_episode_length=max_episode_length)
    return env


def double_reentrant_line_with_demand_model(
        d=1, mu1=4, mu2=3, mu3=2, mu4=3, mu5=4, mus=1e2, mud=1e2,
        cost_per_buffer=np.array([1, 1, 1, 1, 1, 1, 1])[:, None],
        initial_state=np.array([1, 1, 1, 1, 1, 1, 1])[:, None],
        capacity=np.ones((7, 1)) * np.inf,
        job_conservation_flag=False,
        job_gen_seed: Optional[int] = None,
        max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """Double re-entrant line with demand
    As the single re-entrant line with demand but with an extra shared resource.."""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0, 0, mus],
                                         [mu1, -mu2, 0, 0, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0, 0, 0],
                                         [0, 0, mu3, -mu4, 0, 0, 0],
                                         [0, 0, 0, mu4, -mu5, 0, 0],
                                         [0, 0, 0, 0, mu5, -mud, 0],
                                         [0, 0, 0, 0, 0, -mud, 0]])
    constituency_matrix = np.array([[1, 0, 0, 0, 1, 0, 0],
                                    [0, 1, 0, 1, 0, 0, 0],
                                    [0, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 0, 1]])
    demand_rate = np.array([0, 0, 0, 0, 0, 0, d])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 1, 0, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 1, 0, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'
    ind_surplus_buffers = [5]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (4, 3): (3, 3),
                                    (5, 4): (4, 4)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               ind_surplus_buffers=ind_surplus_buffers,
                               max_episode_length=max_episode_length)
    return env


def double_reentrant_line_with_demand_only_shared_resources_model(
        d=1, mu1=4, mu2=3, mu3=3, mu4=4, mus=1e2, mud=1e2,
        cost_per_buffer=np.array([1, 1, 1, 1, 1, 1])[:, None],
        initial_state=np.array([1, 1, 1, 1, 1, 1])[:, None],
        capacity=np.ones((6, 1)) * np.inf, job_conservation_flag=True,
        job_gen_seed: Optional[int] = None, max_episode_length: Optional[int] = None) \
        -> ControlledRandomWalk:
    """Double re-entrant line with demand where all the resources have two activities.
    Figure 7.11 in the printed book."""
    # Rows denote buffer, columns denote influence of activities.
    # Actions mean scheduling processing in one buffer.
    buffer_processing_matrix = np.array([[-mu1, 0, 0, 0, 0, mus],
                                         [mu1, -mu2, 0, 0, 0, 0],
                                         [0, mu2, -mu3, 0, 0, 0],
                                         [0, 0, mu3, -mu4, 0, 0],
                                         [0, 0, 0, mu4, -mud, 0],
                                         [0, 0, 0, 0, -mud, 0]])
    constituency_matrix = np.array([[1, 0, 0, 1, 0, 0],
                                    [0, 1, 1, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1]])
    demand_rate = np.array([0, 0, 0, 0, 0, d])[:, None]
    list_boundary_constraint_matrices = [np.array([[1, 0, 0, 1, 0, 0]]),
                                         np.array([[0, 1, 1, 0, 0, 0]]),
                                         np.array([[0, 0, 0, 0, 1, 0]]),
                                         np.array([[0, 0, 0, 0, 0, 0]])]
    model_type = 'pull'
    ind_surplus_buffers = [4]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (4, 3): (3, 3)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               ind_surplus_buffers=ind_surplus_buffers,
                               max_episode_length=max_episode_length)
    return env


def complex_demand_driven_model(d1=19 / 75, d2=19 / 75,
                                mu1=13 / 15, mu2=26 / 15, mu3=13 / 15, mu4=26 / 15, mu5=1, mu6=2,
                                mu7=1, mu8=2, mu9=1, mu10a=1 / 3, mu10b=1 / 3, mu11=1 / 2,
                                mu12=1 / 10,
                                mud1=100, mus1=100, mud2=100, mus2=100,
                                cost_per_buffer=np.vstack((np.ones((12, 1)),
                                                           np.array([5, 5, 10, 10])[:, None])),
                                initial_state=np.array([10, 5, 8, 3, 4, 4, 6, 5, 6, 7, 2, 2, 0, 0,
                                                        15, 10])[:, None],
                                capacity=np.ones((16, 1)) * np.inf,
                                job_conservation_flag=True,
                                job_gen_seed: Optional[int] = None,
                                max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """ Example 7.2.3. Routing, scheduling, and exogenous demand,
    Figure 7.1 from CTCN book and printed version.
    This setting is used also for several Complex demand-driven example,
    e.g., Example 7.3.2. (CTNT online ed).
    The same network but with different initial states and mud is used in Example 7.2.4.
    of the printed book. """

    # Rows denote buffer, columns denote influence of activities.
    buffer_processing_matrix = np.zeros((16, 17))
    buffer_processing_matrix[0, 0] = -mu1
    buffer_processing_matrix[4, 0] = mu1
    buffer_processing_matrix[1, 1] = -mu2
    buffer_processing_matrix[5, 1] = mu2
    buffer_processing_matrix[2, 2] = -mu3
    buffer_processing_matrix[6, 2] = mu3
    buffer_processing_matrix[3, 3] = -mu4
    buffer_processing_matrix[13, 3] = mu4
    buffer_processing_matrix[4, 4] = -mu5
    buffer_processing_matrix[8, 4] = mu5
    buffer_processing_matrix[5, 5] = -mu6
    buffer_processing_matrix[9, 5] = mu6
    buffer_processing_matrix[6, 6] = -mu7
    buffer_processing_matrix[11, 6] = 0.2 * mu7
    buffer_processing_matrix[12, 6] = 0.8 * mu7
    buffer_processing_matrix[7, 7] = -mu8
    buffer_processing_matrix[2, 7] = mu8
    buffer_processing_matrix[8, 8] = -mu9
    buffer_processing_matrix[3, 8] = mu9
    buffer_processing_matrix[9, 9] = -mu10a
    buffer_processing_matrix[7, 9] = mu10a
    buffer_processing_matrix[9, 10] = -mu10b
    buffer_processing_matrix[10, 10] = mu10b
    buffer_processing_matrix[10, 11] = -mu11
    buffer_processing_matrix[2, 11] = mu11
    buffer_processing_matrix[11, 12] = -mu12
    buffer_processing_matrix[9, 12] = mu12
    buffer_processing_matrix[12, 13] = -mud1
    buffer_processing_matrix[14, 13] = -mud1
    buffer_processing_matrix[13, 14] = -mud2
    buffer_processing_matrix[15, 14] = -mud2
    buffer_processing_matrix[0, 15] = mus1
    buffer_processing_matrix[1, 16] = mus2

    constituency_matrix = np.zeros((9, 17))
    constituency_matrix[0, 0:4] = 1
    constituency_matrix[1, 4:9] = 1
    constituency_matrix[2, 9:11] = 1
    constituency_matrix[3:9, 11:17] = np.eye(6)

    demand_rate = np.vstack((np.zeros((14, 1)), np.array([[d1], [d2]])))
    list_boundary_constraint_matrices = [
        np.hstack((np.ones((1, 4)), np.zeros((1, 12)))),
        np.hstack((np.zeros((1, 4)), np.ones((1, 5)), np.zeros((1, 7)))),
        np.hstack((np.zeros((1, 9)), np.ones((1, 1)), np.zeros((1, 6)))),
        np.hstack((np.zeros((1, 10)), np.ones((1, 1)), np.zeros((1, 5)))),
        np.hstack((np.zeros((1, 11)), np.ones((1, 1)), np.zeros((1, 4)))),
        np.hstack((np.zeros((1, 12)), np.ones((1, 1)), np.zeros((1, 3)))),
        np.hstack((np.zeros((1, 13)), np.ones((1, 1)), np.zeros((1, 2)))),
        np.zeros((1, 16)),
        np.zeros((1, 16))
    ]
    model_type = 'pull'
    ind_surplus_buffers = [12, 13]

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)
    assert job_generator.routes == {(4, 0): (0, 0),
                                    (5, 1): (1, 1),
                                    (6, 2): (2, 2),
                                    (13, 3): (3, 3),
                                    (8, 4): (4, 4),
                                    (9, 5): (5, 5),
                                    (11, 6): (6, 6), (12, 6): (6, 6),
                                    (2, 7): (7, 7),
                                    (3, 8): (8, 8),
                                    (7, 9): (9, 9),
                                    (10, 10): (9, 10),
                                    (2, 11): (10, 11),
                                    (9, 12): (11, 12)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               ind_surplus_buffers=ind_surplus_buffers,
                               max_episode_length=max_episode_length)
    return env


def willems_example_2(resource_capacity=np.array([560000, 70000, 60000, 320000, 40000, 30000,
                                                  1040000, 130000, 120000, 88000, 11000, 10000,
                                                  10000, 300000, 100000, 400000, 40000, 250000]),
                      initial_state=np.ones((17, 1)),
                      capacity=np.ones((17, 1)) * np.inf,
                      job_conservation_flag=True,
                      single_activity_per_resource=True,
                      job_gen_seed: Optional[int] = None,
                      max_episode_length: Optional[int] = None) -> ControlledRandomWalk:
    """
    Willems example 2 ('Optimising Strategic Safety Stock Placement in General Acyclic Networks,
    S. Humair and S.P. Willems, Operation Research 2011, VOl 59, No.3) for the case in which items
    arriving in a buffer from different activities are NOT combined in a single item!!!

    :param initial_state:
    :param capacity:
    :param job_conservation_flag:
    :param job_gen_seed:
    :param max_episode_length:
    :param resource_capacity: 1D array whose elements specify the number of items that can be
        simultaneously processed by a given resources, this can be interpreted as the number of
        items simultaneously moved from one buffer to another when the resource is performing a
        given activity. This divided by the lead time is meant to model the buffer processing rate.
    :param single_activity_per_resource: If true, the version of the example to run is
        'single activity per resource', otherwise it is 'multiple activities per resource'.
    """
    # TODO: Check list bundary constraints to account for safety stock in surplus buffer.
    d1 = 4000
    d2 = 2000
    d3 = 8000
    d4 = 700

    mu1 = resource_capacity[0] / 30
    mu2 = resource_capacity[1] / 15
    mu3 = resource_capacity[2] / 14
    mu4 = resource_capacity[3] / 30
    mu5 = resource_capacity[4] / 15
    mu6 = resource_capacity[5] / 14
    mu7 = resource_capacity[6] / 30
    mu8 = resource_capacity[7] / 15
    mu9 = resource_capacity[8] / 14
    mu10 = resource_capacity[9] / 30
    mu11 = resource_capacity[10] / 15
    mu12 = resource_capacity[11] / 14
    mu13 = resource_capacity[12] / 14
    mud1 = resource_capacity[13] / 5
    mud2 = resource_capacity[14] / 5
    mud3 = resource_capacity[15] / 5
    mud4 = resource_capacity[16] / 5
    mus = resource_capacity[17]

    cost_per_buffer = np.array([80, 80.55, 92.55, 92.55, 80.50, 92.50,
                                92.50, 80.35, 92.35, 184.70, 80.35,
                                92.35, 92.35, 0, 0, 0, 0])[:, None]

    buffer_processing_matrix = np.zeros((17, 18))
    buffer_processing_matrix[0, 0] = -mu1
    buffer_processing_matrix[1, 0] = mu1
    buffer_processing_matrix[1, 1] = -mu2
    buffer_processing_matrix[2, 1] = mu2
    buffer_processing_matrix[2, 2] = -mu3
    buffer_processing_matrix[3, 2] = mu3
    buffer_processing_matrix[0, 3] = -mu4
    buffer_processing_matrix[4, 3] = mu4
    buffer_processing_matrix[4, 4] = -mu5
    buffer_processing_matrix[5, 4] = mu5
    buffer_processing_matrix[5, 5] = -mu6
    buffer_processing_matrix[6, 5] = mu6
    buffer_processing_matrix[0, 6] = -mu7
    buffer_processing_matrix[7, 6] = mu7
    buffer_processing_matrix[7, 7] = -mu8
    buffer_processing_matrix[8, 7] = mu8
    buffer_processing_matrix[8, 8] = -mu9
    buffer_processing_matrix[9, 8] = mu9
    buffer_processing_matrix[0, 9] = -mu10
    buffer_processing_matrix[10, 9] = mu10
    buffer_processing_matrix[10, 10] = -mu11
    buffer_processing_matrix[11, 10] = mu11
    buffer_processing_matrix[11, 11] = -mu12
    buffer_processing_matrix[9, 11] = mu12
    buffer_processing_matrix[11, 12] = -mu13
    buffer_processing_matrix[12, 12] = mu13
    buffer_processing_matrix[3, 13] = -mud1
    buffer_processing_matrix[13, 13] = -mud1
    buffer_processing_matrix[6, 14] = -mud2
    buffer_processing_matrix[14, 14] = -mud2
    buffer_processing_matrix[9, 15] = -mud3
    buffer_processing_matrix[15, 15] = -mud3
    buffer_processing_matrix[12, 16] = -mud4
    buffer_processing_matrix[16, 16] = -mud4
    buffer_processing_matrix[0, 17] = mus

    demand_rate = np.vstack((np.zeros((13, 1)), np.array([[d1], [d2], [d3], [d4]])))
    ind_surplus_buffers = [3, 6, 9, 12]

    if single_activity_per_resource:
        constituency_matrix = np.zeros((15, 18))
        constituency_matrix[0, 0] = 1
        constituency_matrix[0, 3] = 1
        constituency_matrix[0, 6] = 1
        constituency_matrix[0, 9] = 1
        constituency_matrix[1, 1] = 1
        constituency_matrix[2, 2] = 1
        constituency_matrix[3, 4] = 1
        constituency_matrix[4, 5] = 1
        constituency_matrix[5, 7] = 1
        constituency_matrix[6, 8] = 1
        constituency_matrix[7, 10] = 1
        constituency_matrix[8, 11] = 1
        constituency_matrix[9, 12] = 1
        constituency_matrix[10, 13] = 1
        constituency_matrix[11, 14] = 1
        constituency_matrix[12, 15] = 1
        constituency_matrix[13, 16] = 1
        constituency_matrix[14, 17] = 1
        list_boundary_constraint_matrices = [
            np.hstack((np.ones((1, 1)), np.zeros((1, 16)))),
            np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 15)))),
            np.hstack((np.zeros((1, 2)), np.ones((1, 1)), np.zeros((1, 14)))),
            np.hstack((np.zeros((1, 4)), np.ones((1, 1)), np.zeros((1, 12)))),
            np.hstack((np.zeros((1, 5)), np.ones((1, 1)), np.zeros((1, 11)))),
            np.hstack((np.zeros((1, 7)), np.ones((1, 1)), np.zeros((1, 9)))),
            np.hstack((np.zeros((1, 8)), np.ones((1, 1)), np.zeros((1, 8)))),
            np.hstack((np.zeros((1, 10)), np.ones((1, 1)), np.zeros((1, 6)))),
            np.hstack((np.zeros((1, 11)), np.ones((1, 1)), np.zeros((1, 5)))),
            np.hstack((np.zeros((1, 11)), np.ones((1, 1)), np.zeros((1, 5)))),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17))
        ]
    else:
        constituency_matrix = np.zeros((14, 18))
        constituency_matrix[0, 0] = 1
        constituency_matrix[0, 3] = 1
        constituency_matrix[0, 6] = 1
        constituency_matrix[0, 9] = 1
        constituency_matrix[1, 1] = 1
        constituency_matrix[2, 2] = 1
        constituency_matrix[3, 4] = 1
        constituency_matrix[4, 5] = 1
        constituency_matrix[5, 7] = 1
        constituency_matrix[6, 8] = 1
        constituency_matrix[7, 10] = 1
        constituency_matrix[8, 11] = 1
        constituency_matrix[8, 12] = 1
        constituency_matrix[9, 13] = 1
        constituency_matrix[10, 14] = 1
        constituency_matrix[11, 15] = 1
        constituency_matrix[12, 16] = 1
        constituency_matrix[13, 17] = 1
        list_boundary_constraint_matrices = [
            np.hstack((np.ones((1, 1)), np.zeros((1, 16)))),
            np.hstack((np.zeros((1, 1)), np.ones((1, 1)), np.zeros((1, 15)))),
            np.hstack((np.zeros((1, 2)), np.ones((1, 1)), np.zeros((1, 14)))),
            np.hstack((np.zeros((1, 4)), np.ones((1, 1)), np.zeros((1, 12)))),
            np.hstack((np.zeros((1, 5)), np.ones((1, 1)), np.zeros((1, 11)))),
            np.hstack((np.zeros((1, 7)), np.ones((1, 1)), np.zeros((1, 9)))),
            np.hstack((np.zeros((1, 8)), np.ones((1, 1)), np.zeros((1, 8)))),
            np.hstack((np.zeros((1, 10)), np.ones((1, 1)), np.zeros((1, 6)))),
            np.hstack((np.zeros((1, 11)), np.ones((1, 1)), np.zeros((1, 5)))),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17)),
            np.zeros((1, 17))
        ]
    model_type = 'pull'
    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    assert job_generator.routes == {(1, 0): (0, 0),
                                    (2, 1): (1, 1),
                                    (3, 2): (2, 2),
                                    (4, 3): (0, 3),
                                    (5, 4): (4, 4),
                                    (6, 5): (5, 5),
                                    (7, 6): (0, 6),
                                    (8, 7): (7, 7),
                                    (9, 8): (8, 8),
                                    (10, 9): (0, 9),
                                    (11, 10): (10, 10),
                                    (9, 11): (11, 11),
                                    (12, 12): (11, 12)}
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers,
        max_episode_length=max_episode_length
    )
    return env


def tandem_demand_model(n: int,
                        d: int,
                        mu: np.ndarray,
                        cost_per_buffer: np.ndarray,
                        initial_state: np.ndarray,
                        capacity: np.ndarray,
                        job_conservation_flag: Optional[bool] = True,
                        max_episode_length: Optional[int] = None,
                        job_gen_seed: Optional[int] = None,) -> ControlledRandomWalk:
    # TODO: Check list boundary constraints to account for safety stock in surplus buffer.
    """
    Parametric tandem_demand_model example.
    Given the number (n) of blocks composed by buffer-resource-activity,
    it builds a tandem_demand_model composed by n blocks, preceded by a resource for the initial
    supply and followed by a pull demand buffer, a buffer associated with the demand buffer,
    and a fictitious resource for the demand whose demand rate is d.
    """
    assert mu.size == n + 2
    assert len(cost_per_buffer) == n + 2
    assert len(initial_state) == n + 2
    assert len(capacity) == n + 2

    buffer_processing_matrix = np.zeros((n + 2, n + 2))
    for i in range(n):
        buffer_processing_matrix[i, i] = -mu[i]
        buffer_processing_matrix[i + 1, i] = mu[i]
    # buffer_processing for the demand
    buffer_processing_matrix[n, n] = -mu[n]
    buffer_processing_matrix[n + 1, n] = -mu[n]
    # buffer_processing for the supply
    buffer_processing_matrix[0, n + 1] = mu[n + 1]

    constituency_matrix = np.eye(n + 2)

    demand_rate = np.zeros((n + 2, 1))
    demand_rate[n + 1, 0] = d
    ind_surplus_buffers = [n]

    list_boundary_constraint_matrices = []
    for i in range(n):
        constraint = np.zeros((n + 2))
        constraint[i] = 1
        list_boundary_constraint_matrices.append(constraint[None, :])
    # list_boundary_constraint for the demand
    constraint = np.zeros((n + 2))
    list_boundary_constraint_matrices.append(constraint[None, :])
    # list_boundary_constraint for the supply
    constraint = np.zeros((n + 2))
    list_boundary_constraint_matrices.append(constraint[None, :])
    assert len(list_boundary_constraint_matrices) == n + 2

    model_type = 'pull'

    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    routes = {}
    for i in range(n):
        routes[(i + 1, i)] = (i, i)
    assert job_generator.routes == routes
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(
        cost_per_buffer,
        capacity,
        constituency_matrix,
        job_generator,
        state_initialiser,
        job_conservation_flag,
        list_boundary_constraint_matrices,
        model_type,
        ind_surplus_buffers=ind_surplus_buffers,
        max_episode_length=max_episode_length
    )
    return env


def demand_node(alpha=0.33,
                mus=0.34,
                mud=0.99,
                cost_per_buffer=np.array([[1],[10]]),
                initial_state=np.array([[10],[0]]),
                capacity=np.ones((2, 1)) * np.inf,
                job_conservation_flag: Optional[bool] = True,
                max_episode_length: Optional[int] = None,
                job_gen_seed: Optional[int] = None,) -> ControlledRandomWalk:

    buffer_processing_matrix = np.array([[mus,-mud],
                                         [0,-mud]])
    constituency_matrix = np.eye(2)

    demand_rate = np.array([[0],[alpha]])

    model_type = 'pull'
    list_boundary_constraint_matrices = [np.array([[0, 0]]),
                                         np.array([[1, 0]])]
    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    routes: Dict[Tuple, Tuple] = {}
    assert job_generator.routes == routes
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env


def double_demand_node(alpha1=0.33, alpha2=0.33,
                       mus1=0.68, mus2=0.68,
                       mud1=0.9999, mud2=0.9999,
                       cost_per_buffer=np.array([[2],[1],[10],[20]]),
                       initial_state=np.array([[10],[0],[10],[0]]),
                       capacity=np.ones((4, 1)) * np.inf,
                       job_conservation_flag: Optional[bool] = True,
                       max_episode_length: Optional[int] = None,
                       job_gen_seed: Optional[int] = None,) -> ControlledRandomWalk:

    buffer_processing_matrix = np.array([[mus1, 0, -mud1, 0],
                                         [0, mus2, 0, -mud2],
                                         [0, 0, -mud1, 0],
                                         [0, 0, 0, -mud2]])

    constituency_matrix = np.array([[1, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])

    demand_rate = np.array([[0], [0], [alpha1], [alpha2]])

    model_type = 'pull'
    list_boundary_constraint_matrices = [np.array([[0, 0, 0, 0]]),
                                         np.array([[1, 0, 0, 0]]),
                                         np.array([[0, 1, 0, 0]])]
    # Construct environment.
    job_generator = ScaledBernoulliServicesPoissonArrivalsGenerator(demand_rate,
                                                                    buffer_processing_matrix,
                                                                    job_gen_seed=job_gen_seed)

    routes: Dict[Tuple, Tuple] = {}
    assert job_generator.routes == routes
    state_initialiser = DeterministicCRWStateInitialiser(initial_state)
    env = ControlledRandomWalk(cost_per_buffer, capacity, constituency_matrix, job_generator,
                               state_initialiser, job_conservation_flag,
                               list_boundary_constraint_matrices, model_type,
                               max_episode_length=max_episode_length)
    return env
