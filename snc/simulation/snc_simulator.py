import os
import json
from typing import List, Optional, Dict, Any
import numpy as np

from snc.environments import controlled_random_walk as crw
from snc.agents.agent_interface import AgentInterface
import snc.simulation.store_data.reporter as rep
from snc.simulation.store_data.numpy_encoder import NumpyEncoder, format_json_with_np


class SncSimulator:

    def __init__(self, env: crw.ControlledRandomWalk, agent: AgentInterface,
                 **kwargs: Any) -> None:
        """
        Simulates environment and agents, and stores the resulting data.

        :param env: The environment in which to simulate the policy
        :param agent: Agent that gets state and returns actions. It can also perform preliminary
            offline calculations.
        :param **kwargs: set of input parameters that will be stored in parameters.json file.
        """
        assert "discount_factor" in kwargs, "SncSimulator requires a discount factor keyword " \
                                            "argument "
        self.env = env
        self.agent = agent
        self._kwargs = kwargs
        self.discount_factor = kwargs.get("discount_factor")

        # Note: all these objects are available from the `env` object
        # but caching them here, means they are stored as top level parameters during saving.
        self.buffer_processing_matrix = env.job_generator.buffer_processing_matrix
        self.constituency_matrix = env.constituency_matrix
        self.demand_rate = env.job_generator.demand_rate
        self.list_boundary_constraint_matrices = env.list_boundary_constraint_matrices

    def perform_online_simulation(self, num_simulation_steps: int,
                                  reporter: Optional[rep.Reporter] = None,
                                  **override_args) -> Dict[str, List[np.ndarray]]:
        """
        Perform run the online simulations, continuing from current state.

        :param num_simulation_steps: length of data to generate
        :param reporter: a reporter object to document the running of the simulations
        :param override_args: extra keyword arguments to pass to map_state_to_actions
        """
        data = {'state': [], 'action': [], 'zeta_star': [],'cost': [],
                'arrivals': [], 'processing': [],
                'added': [],  'drained': []}  # type: Dict[str, List[np.ndarray]]
        state = self.env.state

        i = 0
        while i < num_simulation_steps:
            action_block = self.agent.map_state_to_actions(state, reporter=reporter,
                                                           **override_args)
            num_mpc_steps = action_block.shape[1]
            for j in range(num_mpc_steps):
                action = action_block[:, j]

                if i >= num_simulation_steps:
                    # do not complete the action block, just return the data
                    break
                new_state, reward, _, extra = self.env.step(action[:, None])

                data['state'].append(state.reshape((state.size,)))
                data['action'].append(action)
                data['zeta_star'].append(getattr(self.agent, 'current_policy', action).ravel())
                data['arrivals'].append(extra['arrivals'].reshape((state.size,)))
                data['processing'].append(extra['processing'].reshape((state.size,)))
                data['added'].append(extra['added'].reshape((state.size,)))
                data['drained'].append(extra['drained'].reshape((state.size,)))
                data['cost'].append(-np.array([reward]))
                state = new_state

                if reporter is not None:
                    reporter.report(data, i)
                i += 1

        return data

    def cache_parameters(self, reporter: Optional[rep.Reporter]) -> None:
        """
        Save all attributes (eg parameters) of the SncSimulator into the cache of the reporter.

        :param reporter: The reporter object logging the simulation.
        """
        if reporter is not None:
            reporter.store(**self.__dict__)

    def run(self, num_simulation_steps: int, reporter: Optional[rep.Reporter] = None,
            save_name: Optional[str] = None, job_gen_seed: Optional[int] = None) \
            -> Dict[str, List[np.ndarray]]:
        """
        Generate state-action data with a policy. First perform offline calculations then run an
        online simulation for num_simulation_steps.

        :param num_simulation_steps: length of data to generate
        :param reporter:  a reporter object to document the running of the simulations
        :param save_name: if specified, save the datadict in the logs directory with same name.
            If it is empty, then nothing is saved.
        :param job_gen_seed: Job generator random seed.
        :return: data - a dictionary mapping to cost, state and action processes
        """
        self.agent.perform_offline_calculations()
        self.cache_parameters(reporter)
        self.env.reset_with_random_state(job_gen_seed)
        data = self.perform_online_simulation(num_simulation_steps, reporter=reporter)

        if save_name is not None:
            self.save(data, save_name, reporter)
        return data

    def save(self, data_dict: Dict[str, List[np.ndarray]], save_name: str,
             reporter: Optional[rep.Reporter] = None) -> None:
        """
        Save the run information as json to a directory in `logs`.

        :param data_dict: a dictionary mapping to cost, state and action processes
        :param save_name: if specified, save the datadict in the logs directory with same name
        :param reporter:  a reporter object to document the running of the simulations
        """
        path = save_name
        reporter_path = '{}/reporter'.format(path)

        if not os.path.isdir(path):
            os.makedirs(reporter_path)

        with open('{}/datadict.json'.format(path), 'w+') as f:
            data_json = json.dumps(data_dict, cls=NumpyEncoder, indent=4, sort_keys=True)
            f.write(format_json_with_np(data_json))
        with open('{}/cost.json'.format(path), 'w+') as f:
            cost_json = json.dumps(data_dict["cost"], cls=NumpyEncoder, indent=4, sort_keys=True)
            f.write(format_json_with_np(cost_json))
        with open('{}/parameters.json'.format(path), 'w+') as f:
            params_json = json.dumps(self.__dict__, cls=NumpyEncoder, indent=4, sort_keys=True)
            f.write(format_json_with_np(params_json))

        if reporter is not None:
            for reporter_item, values_list in sorted(reporter.cache.items()):
                if reporter_item in self.__dict__:
                    continue

                # values_list for reporter_item 'horizon' could be empty.
                # values_list[0] is a namedtuple, so use field names.
                if values_list and hasattr(values_list[0], '_fields'):
                    obj = {f: [getattr(v, str(f)) for v in values_list]
                           for f in values_list[0]._fields}
                else:
                    obj = {reporter_item: values_list}

                with open('{}/{}.json'.format(reporter_path, reporter_item), 'w+') as f:
                    reporter_item_json = json.dumps(obj, cls=NumpyEncoder, indent=4, sort_keys=True)
                    f.write(format_json_with_np(reporter_item_json))
