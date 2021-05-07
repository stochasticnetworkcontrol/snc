from typing import Tuple, List, Set, Optional, Union, Any

import itertools
import numpy as np
import tensorflow as tf
from gym import spaces as gym_spaces

from tf_agents.environments.gym_wrapper import GymWrapper
from tf_agents.environments.tf_py_environment import TFPyEnvironment
from tf_agents.environments.wrappers import ActionRepeat
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment
from tf_agents.environments.suite_gym import wrap_env

import snc.environments.controlled_random_walk as crw
import snc.environments.job_generators.job_generator_interface as jobgen
import snc.environments.state_initialiser as stinit
import snc.utils.snc_tools as snc
import snc.utils.snc_types as snc_types


class RLControlledRandomWalk(crw.ControlledRandomWalk):
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
                 max_episode_length: Optional[int] = None,
                 normalise_observations: bool = True) -> None:
        """
        Environment for testing stochastic network control with reinforcement learning agents:

        This environment is a child of the ControlledRandomWalk class and adds a definition of the
        action space and observation space as required for standard RL implementations.

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
        The episode ends when all buffers are empty.
        Reward is negative (i.e., cost) and linear with the number of jobs in each buffer.
        This is a single agent environment.

        The action space is defined as a series of action sub-spaces one for each set of coupled
        resources. Resources may be coupled by environment constraints. The series of action
        sub-spaces are then independent. The action subspaces are designed to receive one-hot
        actions or a distribution over actions which can then be interpreted by the environment as a
        binary vector of activities for each coupled resource set which can then be added together
        across resource sets to attain an action for the full environment.

        The observation is unbounded and defined as a continuous space of dimension equal to the
        number of buffers. The environment is fully observable where the state is the length of the
        buffers.

        :param cost_per_buffer: cost per unit of inventory per buffer.
        :param capacity: List with maximum number of jobs allowed at each buffer.
        :param constituency_matrix:  Matrix whose s-th row corresponds to resource s; and each
            entry, C_{si} in {0, 1}, specifies whether activity i
            corresponds to resource s (examples of activities are scheduling a buffer or routing to
            another resource).
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
            to None meaning infinite episodes.
        :param normalise_observations: Boolean determining whether the observation of the state is
            normalised such that its L1 norm is unity. This should be set to False for PPO agents as
            they have internal observation normalisation.
        """
        self._is_initialising = True
        super().__init__(
            cost_per_buffer,
            capacity,
            constituency_matrix,
            job_generator,
            state_initialiser,
            job_conservation_flag,
            list_boundary_constraint_matrices,
            model_type,
            index_phys_resources,
            ind_surplus_buffers,
            max_episode_length)

        # The RL environment wrapper augments the initialisation with method calls to construct the
        # action space and construction of the observation space.
        self._construct_action_space()
        self._observation_space = gym_spaces.Box(
            low=0, high=np.inf, shape=(self._num_buffers,), dtype=np.float32)
        self._max_episode_length = max_episode_length
        self._is_initialising = False
        self._normalise_obs = normalise_observations

    def _construct_action_space(self) -> None:
        """
        Builds the action space for an RL agent as a tuple of subspaces one for each independent
        resource set.

        A resource set is a collection of resources for which the action space is coupled due to
        activity constraints. For example, if resources 1 and 2 are constrained such that the action
        of resource 1 could possibly restrict the action space of resource 2 their actions are not
        independent and therefore they form a resource set.
        """
        # Attain the constituency matrix with shadow resources used to invoke constraints removed.
        unconstrained_constituency_matrix = self._constituency_matrix[
            list(self._index_phys_resources)]

        # If there are no constraints to be considered each resource can act independently.
        if len(self._index_phys_resources) == self._num_resources:
            # Resources function independently and can therefore each have their own action space.
            # First work out how many activities are associated with each resource and then the
            # number of actions is the number of possible activities + 1 as activities are mutually
            # independent but there is the possibility of not pursuing any activity (idling).
            actions_per_resource = unconstrained_constituency_matrix.sum(axis=1) + 1
            # Define the action space as a series of binary action spaces.
            action_spaces = list(
                map(lambda d: gym_spaces.Box(low=0.0, high=1.0, shape=(int(d),),
                                             dtype=np.float32), actions_per_resource))
            # We now build a list of action vectors where each resource either idles or pursues one
            # of the activities associated with it.
            action_vectors_list = []
            for resource in range(unconstrained_constituency_matrix.shape[0]):
                # First account for the idle operation
                action_vectors_list.append(np.zeros(self._num_activities))
                # Then find which activities are related to the resource and produce one-hot
                # encodings for them.
                for activity in np.where(unconstrained_constituency_matrix[resource] == 1)[0]:
                    action_vectors_list.append(np.eye(self._num_activities)[activity])
        else:
            # There are cross-resource dependencies to handle making the action space more complex.
            # First attain a matrix which represents the constraints.
            constraint_indices = set(range(self._num_resources)) - set(self._index_phys_resources)
            constraints = self._constituency_matrix[list(constraint_indices)]
            # Ensure that constraints are valid by ensuring that they consider at least two
            # activities. Note that constraints are free to reimpose the mutual exclusivity
            # constraint over activities assumed for each resource. These constraints are
            # unnecessary but valid nonetheless and simply add to the computational load.
            assert np.all(constraints.sum(axis=1) >= 2)
            # Use the constituency matrix and constraints matrix to build the more complex action
            # space with resources coupled by constraints.
            action_spaces, action_vectors_list = self._action_spaces_from_resource_sets(
                unconstrained_constituency_matrix, constraints)
        # Flatten the list of action vectors to get all actions.
        all_action_vectors = action_vectors_list
        self._action_vectors = np.vstack(all_action_vectors)
        # The final action space from an RL point of view is a tuple of the action subspaces for
        # each resource set.
        self._rl_action_space = gym_spaces.Tuple(action_spaces)
        # The action space in terms of activities is a binary space with dimension equal to the
        # number of activities in the network. However, it is implemented as a Box space so that
        # when we wrap the environment for tf_agents the action space takes values of type float as
        # are provided by the neural network.
        self._activities_action_space = gym_spaces.Box(
            low=0.0, high=1.0, shape=(self._num_activities,), dtype=np.float32
        )

    def _action_spaces_from_resource_sets(
            self,
            constituency_matrix: snc_types.ConstituencyMatrix,
            constraints: np.ndarray) -> Tuple[List[gym_spaces.Box], List[np.ndarray]]:
        """
        Calculates the dimension of action spaces for networks where constraints require that some
        resources have dependent action sets. The action spaces are designed to receive one-hot
        actions which correspond to the activities being pursued by the set of resources associated
        with the action subspaces. These actions can then be summed over resource sets to attain an
        action for the full network.

        :param constituency_matrix: The binary matrix representing which activities are associated
            with which resources. Note that this matrix is for physical resources only and hence is
            of shape (num_physical_resources, num_actions)
        :param constraints: The binary matrix representing which actions are constrained not to
            occur together
        :return: A list of action spaces one for each resource set and a list of lists of action
            vectors denoting activities to perform associated with each dimension of the (binary)
            action spaces.
        """
        # Attain a binary matrix representing which resources are affected by which constraints.
        # The resulting binary matrix is of shape (num_resources, num_constraints)
        resource_constraints_matrix = (constituency_matrix @ constraints.T) > 0
        coupled_resource_sets = self._find_coupled_resource_sets(resource_constraints_matrix)
        # Attain one set of all of the resources which cannot be treated independently to aid in
        # checking when to utilise the more complex action space construction code.
        all_coupled_resources = set.union(*coupled_resource_sets)

        # Find which activities are associated with each resource.
        activities_for_resources = np.apply_along_axis(
            lambda ary: set(np.where(ary)[0]),
            axis=1,
            arr=constituency_matrix
        )
        # Initialise empty lists which will be used to build up the action spaces and the associated
        # binary activity tuples which are associated with each action.
        action_spaces = []
        all_action_vectors = []
        # A further set is initialised to store the set of resources which have already been
        # considered to aid in avoiding double counting.
        processed_resources: Set[int] = set()

        # Loop over resources and then build the related action space. This provides a specific
        # ordering so that the action subspaces will always be constructed in the same order and the
        # related resources can be reverse engineered.
        for i in range(constituency_matrix.shape[0]):
            # If we have already considered the resource then do nothing. This will occur when the
            # resource is in fact coupled with a previously processed resource through a constraint.
            if i in processed_resources:
                pass
            elif i in all_coupled_resources:
                # The resource has not been considered yet and is coupled to other resources by at
                # least one constraint.

                # Find which resource set the current resource belongs to and then process the
                # resource set.
                for rs in coupled_resource_sets:
                    if i in rs:
                        # Find the set of possible actions for this resource set by taking the
                        # product of the action spaces for each resource independently and excluding
                        # any actions that violate the constraints.
                        # The action spaces are considered in terms of the activities that each
                        # resource controls. We are careful to include no-activity (idle) as None.
                        # This builds actions as a set of activity indices which are later used to
                        # attain binary action vectors (1 denotes activity is active else 0).

                        # Find all actions (including those which would violate constraints) using
                        # itertools to take the product of the action spaces articulated as activity
                        # indices (as a list `afr` actions for resource). This is a product of the
                        # actions spaces for all resources in the current resource set (`rs`) where
                        # each resource is treated as though it may act independently. The actions
                        # which violate constraints are removed later. `None` is used to denote
                        # idling since there is no valid index for idling as this is a vector of
                        # zeros when articulated in the binary action space (of dimension
                        # `num_activities`).
                        all_actions = [*itertools.product(*[[None] + list(afr) for afr
                                                            in activities_for_resources[list(rs)]])]
                        # Initialise a list of valid action vectors.
                        action_vectors = []
                        # Consider each possible action, encode it as a binary action and determine
                        # whether or not it is valid.
                        for action in all_actions:
                            # Determine which activities are active under each action by filtering
                            # out Nones. This then provides the indices to set to 1 in the binary
                            # action vector.
                            activities = list(filter(lambda a: a is not None, action))
                            # Set up the binary action vector
                            binary_action = np.zeros(self._num_activities)
                            if activities:
                                binary_action[activities] = 1
                            # Determine whether the action violates any constraints. It violates a
                            # constraint if at least two of the activities in the constraint are
                            # being pursued since the constraints impose that at most one of the
                            # activities they mark can be active at one time.
                            if not np.any(constraints.dot(binary_action) >= 2):
                                # If the action is valid append it to the list of action vectors.
                                action_vectors.append(binary_action)
                        # Add the list of possible actions for this resource set to the list of
                        # action sets.
                        all_action_vectors.append(
                            np.reshape(np.array(action_vectors), (-1, self._num_activities))
                        )
                        # Set up the gym action space using the fact that each dimension of the gym
                        # action space corresponds to one of the action vectors. The gym space is a
                        # Box so that when we wrap the environment for tf_agents the action space
                        # takes values of type float as are provided by the neural network.
                        num_actions = len(action_vectors)
                        action_spaces.append(
                            gym_spaces.Box(
                                low=0.0, high=1.0, shape=(num_actions,), dtype=np.float32
                            )
                        )
                        # Ensure that we mark the resources in the current set as processed.
                        processed_resources = processed_resources.union(rs)
            else:
                # The resource can act independently of all others and therefore the action space
                # is relatively simple.
                # The number of actions is equal to the number of activities associated with the
                # resource + 1 as each activity is mutually exclusive and it is also possible to do
                # nothing.
                num_actions = int(constituency_matrix[i].sum() + 1)
                # action_vectors is a list of binary vectors denoting which activities are being
                # taken.
                action_vectors = [np.zeros(self._num_activities)]
                # To build action vectors we need to find which activities this resource can
                # possibly undertake.
                possible_activities = np.where(constituency_matrix[i] == 1)[0]
                # The possible activities then form actions when represented by one-hot encodings
                # in this case (as we only consider one resource so one activity with idle already
                # included).
                action_vectors += [np.eye(self._num_activities)[k] for k in possible_activities]
                # Add the list of action vectors to the list of lists of action vectors for each
                # resource set.
                action_vector_array = np.array(action_vectors)
                all_action_vectors.append(
                    np.reshape(action_vector_array, (-1, self._num_activities))
                )
                # Finally set up the gym action space where each dimension corresponds to one action
                # vector.
                # The gym space is a Box so that when we wrap the environment for tf_agents the
                # action space takes values of type float as are provided by the neural network.
                action_spaces.append(
                    gym_spaces.Box(low=0.0, high=1.0, shape=(num_actions,), dtype=np.float32))
        return action_spaces, all_action_vectors

    @staticmethod
    def _find_coupled_resource_sets(resource_constraints_matrix: np.ndarray) -> List[set]:
        """
        Determines which resources are coupled by the constraints. This is done by initially finding
        which resources are affected by each constraint and then find whether there are chains in
        the constraints which link multiple resource sets.

        e.g. constraint 1 affects resources 1 and 2, constraint 2 affects resources 3 and 4 and
        constraint 3 affects resources 1 and 4 such that overall resources 1, 2, 3 and 4 have to
        be treated jointly.

        :param resource_constraints_matrix:  Binary matrix representing which resources are affected
            by which constraints. This matrix should have shape (num_resources, num_constraints).
        :return: A list of sets of resource sets where resources are grouped according to the
            constraints which cause their activities (and therefore their action spaces) to be
            dependent.
        """
        # Find which resources are linked by each constraint (without then considering the deeper
        # links occurring from multiple constraints).
        resource_sets = np.apply_along_axis(
            lambda ary: set(np.where(ary)[0]),
            axis=0,
            arr=resource_constraints_matrix
        )
        # Initialise a list which will ultimately contain sets of resources which are grouped.
        combined_resource_sets: List[Set[int]] = []
        # Consider each resource set from the initial shallow parsing of constraints.
        for rs in resource_sets:
            # If there is already at least one group of resources then we must consider if the new
            # resource set links them.
            if combined_resource_sets:
                # Initialise a resource set which combines resources coupled by multiple
                # constraints.
                combined_rs = rs
                # Initialise a list to keep track of the indices of the existing resource sets which
                # will be superseded by the new larger set so that they can be removed from the list
                # of resource sets later.
                rs_indices = []

                # Consider each existing resource set and see whether it needs to be merged with the
                # resource set currently being considered.
                for i, existing_rs in enumerate(combined_resource_sets):
                    # The resource sets need to be merged if the same resource appears in both.
                    if existing_rs.intersection(combined_rs):
                        # Track the index of the existing resource set which has been merged and is
                        # therefore now obsolete.
                        rs_indices.append(i)
                        # Merge the resource sets
                        combined_rs = combined_rs.union(existing_rs)
                # Now remove all of the resource sets which have been merged so that no resource
                # appears in more than one resource set. Note that we do this from the largest index
                # to the smallest to avoid messing with the indexing for later removals.
                for i in reversed(rs_indices):
                    combined_resource_sets.pop(i)
                # Finally add the enlarged resource set to the list of resource sets.
                combined_resource_sets.append(combined_rs)
            else:
                # There are no other resource sets to consider yet so simply add this resource set
                # as the first one in the list.
                combined_resource_sets.append(rs)
        return combined_resource_sets

    def _interpret_rl_action(self, binary_action: np.ndarray) -> np.ndarray:
        """
        Takes an action from the RL agent and puts it in terms of activities in the network.

        :param binary_action: The action vector(s) from the RL agent which is formed from the
            concatenation of the one-hot actions for each resource set.
        :return: A binary action vector where the 1s denote active activities.
        """
        # Ensure that the action resulting from the RL agent is binary.
        # Binarisation should be handled in the agent.
        assert snc.is_binary(binary_action)
        # Determine which action vectors are selected.
        action_indices = np.where(binary_action == 1)[-1]
        # Use action indices to attain constituent action vectors (for each resource set)
        action_vectors = self._action_vectors[action_indices]
        # The overall action combines the activities across resource sets. This can be achieved by
        # summing over resource sets
        action = np.sum(action_vectors, axis=0)
        # Check that the action is binary (i.e. has no entries outside of {0, 1} which could occur
        # if the action vectors are not set up correctly)
        assert snc.is_binary(action)
        return action

    def preprocess_action(
            self,
            action: Union[np.ndarray, tf.Tensor, Tuple[Union[tf.Tensor, tf.Variable, np.ndarray]]]
    ) -> np.ndarray:
        """
        Prepares the action(s) for the original environment by putting them in the right format and
        transforming them from a series of one-hot vectors per resource set to a single binary
        action vector where 1 denotes active activities and 0 denotes idle activities.

        :param action: The actions for the resource sets in the environment as a numpy array or
            TensorFlow tensor if there is only one resource set or a tuple of one-hot vectors (one
            for each resource set).
        :return: The binary action as a numpy array of shape (num_activities, 1)
        """
        # If a tuple is provided but there is only one resource set then we can simply take the
        # action out of the tuple.
        if isinstance(action, tuple) and len(action) == 1:
            action = action[0]
        # Otherwise we need to concatenate actions across resource sets to attain an over action for
        # the network. This is done carefully accounting for different concatenation approaches in
        # numpy and TensorFlow.
        if isinstance(action, tuple):
            if isinstance(action[0], np.ndarray):
                action = np.concatenate(action, axis=-1)
            # If the actions for each subspace are valid TensorFlow 2 objects they will have a numpy
            # method to attain a numpy array from the Tensor/Variable.
            elif all([hasattr(a, "numpy") for a in action]):
                action = tf.concat(action, axis=-1).numpy()
            else:
                # Actions must be supplied as numpy arrays or TensorFlow tensors.
                raise ValueError(
                    "Action needs to be a (tuple of) valid TensorFlow or numpy objects."
                )
        # If the action is in a TensorFlow format attain it as a numpy array so that the environment
        # need only handle numpy arrays.
        elif hasattr(action, "numpy"):
            action = action.numpy()  # type: ignore
        # In the case that the action is already in the correct format then simply pass it through.
        elif isinstance(action, np.ndarray):
            pass
        else:
            # Catch any malformed actions (in terms of type).
            raise ValueError("Action needs to be a (tuple of) valid TensorFlow or numpy objects.")
        # Use the _interpret_rl_action method to transform the set of one-hot actions to a binary
        # action in the format required.
        action_in_activity_space = self._interpret_rl_action(action)
        # The environment expects column vectors so reshape if needed.
        if len(action_in_activity_space.shape) == 1:
            action_in_activity_space = action_in_activity_space[:, None]
        return action_in_activity_space

    def step(
            self,
            action: Union[np.ndarray, Tuple[np.ndarray]]
    ) -> Tuple[snc_types.StateSpace, float, bool, Any]:
        """
        Processes the action from the agent and then returns the environment state, reward,
        termination flag and information from the environment.

        :param action: The actions for the resource sets in the environment as a numpy array or
            TensorFlow tensor, if there is only one resource set or a tuple of one-hot vectors (one
            for each resource set).
        :return: next_state, reward, done, info
        """
        # Process the action(s) provided and then pass them on to the step method of the underlying
        # ControlledRandomWalk class.
        action = self.preprocess_action(action)
        state, reward, done, info = super().step(action)
        # Scale the state as it is used as an input to a neural net.
        scaled_state = self.normalise_state(state)
        # Transpose the state to ensure that is (batch_size x num_buffers) as required for
        # TensorFlow.
        # At least 1D used to ensure that in the case of a 1 dimensional state an array is returned
        # rather than a scalar.
        return np.atleast_1d(np.squeeze(scaled_state.T)), reward, done, info

    def normalise_state(self, state: snc_types.StateSpace) -> snc_types.StateSpace:
        """
        Normalises the observed state by dividing by the sum of the entries to make it more amenable
        to neural networks.

        :param state: The unscaled state from the ControlledRandomWalk instance.
        :return: The scaled state.
        """
        scaled_state = state.astype(np.float32)
        if self._num_buffers > 1 and scaled_state.sum() > 0:
            scaled_state = scaled_state / scaled_state.sum()
        return scaled_state

    def reset(self) -> snc_types.StateSpace:
        """
        Wraps the reset method of the original environment to manipulate the returned state such
        that it is of a form amenable to TensorFlow (shape batch_size x observation_dimension)

        :return: numpy array of the current state.
        """
        state = super().reset()

        # Handle the initial call from super().__init__
        if self._is_initialising:
            return state

        # Scale the state as it is used as an input to a neural net.
        if self._normalise_obs:
            scaled_state = self.normalise_state(state)
        else:
            scaled_state = state.astype(np.float32)

        # At least 1D used to ensure that in the case of a 1 dimensional state an array is
        # returned rather than a scalar.
        return np.atleast_1d(np.squeeze(scaled_state.T))

    @property
    def activities_action_space(self):
        return self._activities_action_space

    @property
    def action_space(self):
        return self._rl_action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_vectors(self):
        return self._action_vectors

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def normalise_obs(self):
        return self._normalise_obs

    @normalise_obs.setter
    def normalise_obs(self, normalise: bool):
        self._normalise_obs = normalise

    def render(self, mode='human'):
        raise NotImplementedError


def rl_env_from_snc_env(env: crw.ControlledRandomWalk, discount_factor: float,
                        tf_action_repetitions: int = 1, parallel_environments: int = 1,
                        for_tf_agent: bool = True, normalise_observations: bool = True) \
        -> Tuple[Union[TFPyEnvironment, RLControlledRandomWalk], Tuple[int, ...]]:
    """
    Wraps a standard environment to make a TensorFlow compatible environment.

    :param env: The original ControlledRandomWalk environment.
    :param discount_factor: The discount factor applied to future returns.
    :param tf_action_repetitions: Number of time steps each selected action is repeated for.
        TensorFlow environment only.
    :param parallel_environments: Number of environments to run in parallel.
    :param for_tf_agent: Boolean determining whether to wrap the ControlledRandomWalk to be fully
        TensorFlow compatible (defaults to True). This may be false in cases where we just want
        to augment the environment with RL action interpretation methods.
    :param normalise_observations: Boolean determining whether the observation of the state is
            normalised such that its L1 norm is unity.
    :return: A wrapped environment ready for Reinforcement Learning/TensorFlow Agents alongside
        a tuple describing the action space dimensions.
    """
    assert isinstance(tf_action_repetitions, int), 'Action repetitions must be an integer.'
    assert tf_action_repetitions > 0, 'Each action must be repeated a positive number of times.'
    # First initialise an RLControlledRandomWalk as defined above.
    base_env = RLControlledRandomWalk(
        cost_per_buffer=env.cost_per_buffer,
        capacity=env.capacity,
        constituency_matrix=env.constituency_matrix,
        job_generator=env.job_generator,
        state_initialiser=env.state_initialiser,
        job_conservation_flag=env.energy_conservation_flag,
        list_boundary_constraint_matrices=env.list_boundary_constraint_matrices,
        model_type=env.model_type,
        index_phys_resources=env.index_phys_resources,
        max_episode_length=env.max_episode_length,
        normalise_observations=normalise_observations
    )
    action_space_dims = tuple(space.shape[-1] for space in base_env.action_space.spaces)

    # Test that the action dimensions aggregate over the network to provide a valid full action set.
    assert sum(action_space_dims) == len(base_env.action_vectors)

    if not for_tf_agent:
        return base_env, action_space_dims

    def env_constructor(rl_env, seed=1):
        """Function used to set up new environments for each process when multiprocessing."""
        rl_env.reset_with_random_state(seed)
        # Wrap the RLControlledRandomWalk ready for a Python implementation of a TFAgent
        # (using GymWrapper) and wrap for action repetitions if necessary.
        if tf_action_repetitions > 1:
            rl_env = ActionRepeat(GymWrapper(rl_env, discount=discount_factor),
                                  tf_action_repetitions)
        else:
            rl_env = GymWrapper(rl_env, discount=discount_factor)
        return rl_env

    # Set up the parallel environments from a list of constructors and wrap the result ready for
    # a TensorFlow agent.
    env_constructors = [lambda j=i: env_constructor(base_env,
                                                    (base_env.job_generator.seed or 0) + j)
                        for i in range(parallel_environments)]
    parallel_env = TFPyEnvironment(ParallelPyEnvironment(env_constructors))
    return parallel_env, action_space_dims
