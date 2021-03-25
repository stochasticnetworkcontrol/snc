import numpy as np

from typing import Dict, Any
from src.snc.utils import snc_types

from tf_agents.agents import PPOAgent
from tf_agents.agents.tf_agent import TFAgent
from tf_agents.trajectories.time_step import TimeStep, StepType

from src.snc import AgentInterface
from src.snc.environments import \
    ControlledRandomWalk
from src.snc.environments import \
    rl_env_from_snc_env


class RLSimulationAgent(AgentInterface):
    """Wrapper for a TensorFlow agent to ensure compatibility with the SNC simulator."""

    def __init__(self, env: ControlledRandomWalk, agent: TFAgent, normalise_obs: bool,
                 name: str = "RLSimulationAgent", evaluation: bool = False):
        """
        Sets up the simulation agent from an environment and a standard TensorFlow Agent.

        Note: The environment is not used for simulation, simply for interpreting RL Agent actions.

        :param env: The SNC environment for the simulation.
        :param agent: The fully initialised (and trained) TensorFlow agent.
        :param name: Agent identifier.
        :param evaluation: Determines whether to use the greedy policy or not. Defaults to True i.e.
            use greedy policy.
        """
        # Attain an RLControlledRandomWalk instance from the ControlledRandomWalk provided.
        # This is used to interpret actions from an RL Agent.
        self.discount_factor = agent._discount_factor if isinstance(agent, PPOAgent) \
            else agent._gamma
        self._rl_env, _ = rl_env_from_snc_env(
            env,
            discount_factor=self.discount_factor,
            for_tf_agent=False,
            normalise_observations=normalise_obs
        )

        # Set up private properties required for map_state_to_actions
        self._rl_agent = agent
        self._is_eval_policy = evaluation
        self._normalise_obs = normalise_obs
        self._policy = self._rl_agent.policy if self._is_eval_policy \
            else self._rl_agent.collect_policy

        # Call the standard initialiser.
        super().__init__(env, name)

    def map_state_to_actions(self, state: snc_types.StateSpace, **override_args: Any) \
            -> snc_types.ActionProcess:
        """
        The action function taking in the observed state and returning an action vector.

        :param state: The observed state of the environment.
        :param override_args: Dictionary of additional keyword arguments (in this case all
            additional arguments are ignored).
        :return: An action vector in the format expected by the SNC simulator.
        """
        # To be compatible with the TensorFlow agent we must form a TimeStep object to pass the data
        # to the agent.
        # Note that for step types 0 is the initial time step, 1 is any non-terminal time step state
        # and 2 represents the final time step.
        # The reward is not provided. This code will need to be refactored if the agent's decision
        # making is based on the reward.
        if self.env.is_at_final_state:
            step_type = StepType(2)
        else:
            step_type = StepType(int(1 - self.env.is_at_initial_state))

        # Scale the state as is done in RLControlledRandomWalk in order to allow RL to work.
        if self._normalise_obs:
            scaled_state = self._rl_env.normalise_state(state)
        else:
            scaled_state = state.astype(np.float32)

        time_step = TimeStep(
            step_type=step_type,
            reward=None,
            discount=override_args.get("discount_factor", self.discount_factor),
            observation=scaled_state.reshape(1, state.shape[0])
        )
        # The action provided by the TensorFlow agent is in a form suitable for the TensorFlow
        # environment. We therefore use the environment's action processing method to convert the
        # action in to a suitable form for the SNC simulator.
        rl_action = self._policy.action(time_step)
        snc_action = self._rl_env.preprocess_action(rl_action.action)
        return snc_action

    def to_serializable(self) -> Dict:
        """
        Return a serializable object, that can be used by a JSON encoder.
        """
        # To avoid issues when trying to serialise TensorFlow objects they are replaced with strings
        # which can be serialised.
        as_dict = {
            "env": self.env,
            "buffer_processing_matrix": self.buffer_processing_matrix,
            "constituency_matrix": self.constituency_matrix,
            "demand_rate": self.demand_rate,
            "list_boundary_constraint_matrices": self.list_boundary_constraint_matrices,
            "name": self.name,
            "_policy": str(self._policy),
            "_is_eval_policy": self._is_eval_policy,
            "_rl_env": str(self._rl_env),
            "_rl_agent": str(self._rl_agent)
        }
        return as_dict

    def __str__(self):
        """Refined string conversion to differentiate between instances when logging."""
        return f"{self.__class__} - Instance Name {self.name}"
