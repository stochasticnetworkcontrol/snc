from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import gym
import numpy as np

import src.snc.environments.job_generators.job_generator_interface as jobgen
import src.snc.environments.state_initialiser as stinit
from src import snc as snc, snc as snc_types


class ControlledRandomWalk(gym.Env):

    def __init__(self,
                 cost_per_buffer: snc_types.StateSpace,
                 capacity: snc_types.StateSpace,
                 constituency_matrix: snc_types.ConstituencyMatrix,
                 job_generator: jobgen.JobGeneratorInterface,
                 state_initialiser: stinit.CRWStateInitialiser,
                 job_conservation_flag: bool = False,
                 list_boundary_constraint_matrices: Optional[List[np.ndarray]] = None,
                 model_type: str = 'push',
                 index_phys_resources: Optional[Tuple] = None,
                 ind_surplus_buffers: Optional[List[int]] = None,
                 max_episode_length: Optional[int] = None) -> None:
        """
        Environment for testing stochastic network control:

        We consider a network of (possibly) multiple resources, each one with (possibly) multiple
        buffers. The resources can schedule in which of its buffers they process jobs at every time
        step, and where to route the processed jobs.

        The controlled random walk (CRW) describes the stochastic evolution of jobs in each
        buffer, under some policy that controls how the resources schedule and route the jobs,
        and where the arrival of new jobs and the processing of the current jobs are stochastic.
        The arrivals follow a job generator, which could be e.g. a Poisson process, such that the
        number of new jobs at each time step follows a Poisson distribution, with mean given by
        some demand rate vector. The success of finishing a job when a resource is working on it
        is also given by the job generator, which could be e.g. another Poisson process.

        Each buffer can start filled with some amount of jobs.
        Reward is negative (i.e., cost) and linear with the number of jobs in each buffer.
        This is a single agent environment.

        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param capacity: List with maximum number of jobs allowed at each buffer.
        :param constituency_matrix: Matrix whose s-th row corresponds to resource s; and each entry,
            C_{si} in {0, 1}, specifies whether activity i corresponds to resource s (examples of
            activities are scheduling a buffer or routing to another resource).
        :param job_generator: object to generate events from
        :param state_initialiser: initialiser for state
        :param job_conservation_flag: boolean flag that controls whether:
            'True' = we want to ensure energy conservation when updating the step (moving only jobs
            that exist), or 'False' = We leave this task to the policy. This parameter is False by
            default.
        :param list_boundary_constraint_matrices: List of binary matrices, one per resource, that
            indicates conditions (number of rows) on which buffers cannot be empty to avoid idling.
            If this parameter is not passed, it assumes no boundary constraints by default.
        :param model_type: String indicating if this is a `'pull'` or `'push'` model.
        :param index_phys_resources: Tuple indexing the rows of constituency matrix that correspond
            to physical resources (as opposed to other rows that might represent coupled constraints
            among physical resources). If this parameter is not passed, then it considers all rows
            of the constituency matrix as corresponding to physical resources by default.
        :param ind_surplus_buffers: List of integers indicating the location of the surplus
            buffers in the buffer processing matrix.
        :param max_episode_length: Integer number of time steps to allow in each episode. Defaults
            to None meaning unlimited steps per episode (i.e. non-terminating episodes).
        """

        assert np.all(cost_per_buffer >= 0), 'Cost per buffer is assumed nonnegative'
        assert np.all(capacity >= 0), 'Capacity must be nonnegative'
        assert snc.is_binary(constituency_matrix), 'Constituency matrix must be binary'

        self._num_buffers = cost_per_buffer.size
        assert cost_per_buffer.shape == (self.num_buffers, 1)
        assert capacity.shape == (self.num_buffers, 1)

        self._num_resources, self._num_activities = constituency_matrix.shape
        assert job_generator.buffer_processing_matrix.shape == (self.num_buffers,
                                                                self.num_activities)
        assert job_generator.num_buffers == self.num_buffers

        assert model_type in ['push', 'pull']

        self._cost_per_buffer = cost_per_buffer
        self._capacity = capacity
        self._constituency_matrix = constituency_matrix
        self._job_generator = job_generator
        self._state_initialiser = state_initialiser
        self._job_conservation_flag = job_conservation_flag
        self._model_type = model_type

        # If no list_boundary_constraint_matrices is passed, initialise to empty.
        if list_boundary_constraint_matrices is None:
            self._list_boundary_constraint_matrices = []  # type: List[np.ndarray]
            for s in range(self.num_resources):
                self._list_boundary_constraint_matrices.append(np.array([]))
        else:
            assert len(list_boundary_constraint_matrices) == self.num_resources
            for s in range(self.num_resources):
                assert list_boundary_constraint_matrices[s].shape[1] == self.num_buffers
            self._list_boundary_constraint_matrices = list_boundary_constraint_matrices

        # If no index_phys_resources is passed, assume all rows of constituency matrix are physical
        # resources.
        if index_phys_resources is None:
            self._index_phys_resources = tuple(range(self._num_resources))
        else:
            assert min(index_phys_resources) >= 0
            assert max(index_phys_resources) < self._num_resources  # index in [0, num_resources -1]
            assert len(set(index_phys_resources)) == len(index_phys_resources)
            self._index_phys_resources = index_phys_resources

        if model_type == 'pull':
            assert ind_surplus_buffers is not None, \
                "Pull models require surplus buffers, but none has been provided."
            assert isinstance(ind_surplus_buffers, list), \
                f"Type of 'ind_surplus_buffers' must be list, but provided: " \
                f"{type(ind_surplus_buffers)}."
            ind_surplus_buffers.sort()
            assert self.is_surplus_buffers_consistent_with_job_generator(
                ind_surplus_buffers,
                self.job_generator.demand_nodes.keys()
            ), "Provided list of surplus buffers is not consistent with buffer processing matrix."
        self._ind_surplus_buffers = ind_surplus_buffers

        # Build constituency with rows corresponding to physical resources.
        self._physical_constituency_matrix = constituency_matrix[self.index_phys_resources, :]

        # Set the episode length parameter and time step counter
        self._max_episode_length = max_episode_length
        self._t = 0

        # Set up the state attribute ahead of it being set in the reset method.
        self.state = self.state_initialiser.get_initial_state()
        self.reset()

    def to_serializable(self) -> Dict:
        """Return a serializable object, that can be used by a JSON Encoder"""
        return deepcopy(self.__dict__)

    @property
    def cost_per_buffer(self):
        return self._cost_per_buffer

    @property
    def capacity(self):
        return self._capacity

    @property
    def constituency_matrix(self):
        return self._constituency_matrix

    @property
    def job_generator(self):
        return self._job_generator

    @job_generator.setter
    def job_generator(self, new_job_generator):
        self._job_generator = new_job_generator

    @property
    def state_initialiser(self):
        return self._state_initialiser

    @property
    def energy_conservation_flag(self):
        return self._job_conservation_flag

    @property
    def list_boundary_constraint_matrices(self):
        return self._list_boundary_constraint_matrices

    @property
    def model_type(self):
        return self._model_type

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def num_resources(self):
        return self._num_resources

    @property
    def num_activities(self):
        return self._num_activities

    @property
    def index_phys_resources(self):
        return self._index_phys_resources

    @property
    def ind_surplus_buffers(self):
        return self._ind_surplus_buffers

    @property
    def physical_constituency_matrix(self):
        return self._physical_constituency_matrix

    @property
    def max_episode_length(self):
        """
        The maximum length of an episode. Required when using the environment for RL.
        At the time step max_episode_length + 1 the environment resets to its initial random state.
        If max_episode_length is None episodes are infinite, see ControlledRandomWalk.step.
        """
        return self._max_episode_length

    @max_episode_length.setter
    def max_episode_length(self, v):
        self._max_episode_length = v

    @property
    def t(self) -> int:
        return self._t

    @property
    def is_at_initial_state(self) -> bool:
        if self.max_episode_length is not None:
            return self.t % self.max_episode_length == 0
        else:
            return self.t == 0

    @property
    def is_at_final_state(self) -> bool:
        if self.max_episode_length is not None:
            return self.t % self.max_episode_length == self.max_episode_length - 1
        else:
            return False

    @staticmethod
    def is_surplus_buffers_consistent_with_job_generator(
            ind_surplus_buffers: List[int],
            surplus_nodes: List[Tuple[int]]) -> bool:
        """
        Checks the passed surplus buffer indexes are consistent with the surplus buffers detected
        in the buffer processing matrix by the job generator.

        :param ind_surplus_buffers: Tuple of integers representing the indexes of the surplus
            buffers passed when describing the environment.
        :param surplus_nodes: List of integers representing the indexes of the surplus buffers
            detected by the job generator.
        :return: Boolean indicating if passed surplus buffers are consistent with job generator.
        """
        surplus_buffers = sorted([b[0] for b in surplus_nodes])
        return sorted(ind_surplus_buffers) == surplus_buffers

    def ensure_jobs_conservation(
            self,
            routing_jobs_matrix: snc_types.BufferMatrix,
            state_plus_arrivals: snc_types.StateSpace
    ) -> snc_types.BufferMatrix:
        """
        If the number of serviced jobs in the routing_jobs_matrix is more than the number of
        items that are available in the buffers, then the routing_jobs_matrix is truncated.
        The process is done in a sequence of steps:
          1. We assume that new arrivals have arrived before truncating
          2. We truncate jobs being routed if needed, and ensure that the number of items leaving
             origin and arriving at destination are equal.
          3. We truncate the amount of demand being satisfied if needed in pull models, and ensure
             that the number of items removed from surplus are deficit buffers are equal.
          4. We truncate the amount of items leaving the system if needed in push models.

        The order of this sequence is somehow arbitrary and could influence the result though.

        :param routing_jobs_matrix:
        :param state_plus_arrivals:
        :return Adjusted routing_jobs_matrix.
        """
        # For routing, set number of jobs leaving a buffer equal to the number jobs that input the
        # next buffer in the route.
        # We create a temporary state to be able to update what is actually in the buffers
        temp_state = state_plus_arrivals.copy()
        for destination, origin in self.job_generator.routes.items():
            # Check that we don't move more jobs than what there is actually in the buffers.
            available_jobs = np.min((temp_state[origin[0]], np.abs(routing_jobs_matrix[origin])))
            routing_jobs_matrix[origin] = - available_jobs  # Negative, leaves origin.
            routing_jobs_matrix[destination] = available_jobs  # Positive, gets into destination.
            # We update the temporary state so that if there are multiple routes that leave the same
            # origin then we don't move more jobs than what there is actually in the buffers.
            temp_state[origin[0]] -= available_jobs

        # For demand nodes, same amount is retired from both surplus and deficit buffers.
        for surplus, deficit in self.job_generator.demand_nodes.items():
            # Check that we don't move more jobs than what there is actually in the buffers.
            available_jobs = np.min((temp_state[surplus[0]], np.abs(routing_jobs_matrix[surplus]),
                                     temp_state[deficit[0]], np.abs(routing_jobs_matrix[deficit])))
            routing_jobs_matrix[surplus] = - available_jobs
            routing_jobs_matrix[deficit] = - available_jobs

        for e in self.job_generator.exit_nodes:
            # Check that we don't move more jobs than what there is actually in the buffers.
            available_jobs = np.min((temp_state[e[0]], np.abs(routing_jobs_matrix[e])))
            routing_jobs_matrix[e] = - available_jobs

        return routing_jobs_matrix

    def step(self, action: snc_types.ActionSpace) \
            -> Tuple[snc_types.StateSpace, float, bool, Any]:
        """
        Implements controlled random walk. We remove jobs from each buffer if it is scheduled,
        and adds a new job if it has arrived any. Arrivals, processing and routing events happen
        all at the same time.

        :param action: Current action performed by the agent.
        :return: (state, reward, done, extra_data):
            - state: New state after performing action in current state.
            - reward: Reward (negative cost) obtained after the state transition.
            - done: Indicates if maximum number of steps has been reached. Currently used for
                training RL agents. It can implement other termination conditions in the future.
            - extra_data: Dictionary including new arrivals, total number of precessed jobs,
                processed jobs that have been added to any buffer (from suppliers or routed from
                other buffers), and processed jobs that have been drained from any buffer.
        """
        assert snc.is_binary(action)
        assert action.shape == (self.num_activities, 1)
        assert np.all(self.constituency_matrix @ action <= 1), \
            "Current action violates one action per resource constraint: C u <= 1."

        new_jobs = self.job_generator.get_arrival_jobs()  # New job arrivals per buffer
        # Generate number of processed jobs
        routing_jobs_matrix = self.job_generator.get_routing_job_matrix()
        if self.energy_conservation_flag:
            routing_jobs_matrix = self.ensure_jobs_conservation(routing_jobs_matrix,
                                                                self.state + new_jobs)
        action_effects = np.dot(routing_jobs_matrix, action)
        additions_effects = np.dot(np.clip(routing_jobs_matrix, a_min=0, a_max=None), action)
        drained_effects = np.dot(np.clip(routing_jobs_matrix, a_min=None, a_max=0), action)

        self.state = self.state + new_jobs + action_effects  # Update CRW state

        # Ensure physical constraints are satisfied.
        assert np.all(self.state >= 0)
        assert np.all(self.state <= self.capacity)

        cost = np.dot(self.cost_per_buffer.transpose(), self.state)
        reward = - float(cost)

        extra_data = {
            'arrivals': new_jobs,
            'processing': action_effects,
            'added': additions_effects,
            'drained': drained_effects
        }

        # Increment the time step.
        self._t += 1

        done = False
        if self.max_episode_length is not None and self._t >= self.max_episode_length:
            done = True
        return self.state, reward, done, extra_data

    def reset(self) -> snc_types.StateSpace:
        """
        Resets buffers to their initial states according to the initialiser,
        which can be deterministic, random.
        """
        self.state = self.state_initialiser.get_initial_state()
        self._t = 0
        return self.state

    def reset_with_random_state(self, job_gen_seed: Optional[int] = None) -> snc_types.StateSpace:
        """
        Runs the environment reset while also resetting the environment's random state. This resets
        to the initial random seed unless a seed is provided.
        :param job_gen_seed: Job generator random seed.
        :return: The initial state.
        """
        self.job_generator.reset_seed(job_gen_seed)
        return self.reset()

    def render(self, mode='human'):
        raise Exception(NotImplemented)
