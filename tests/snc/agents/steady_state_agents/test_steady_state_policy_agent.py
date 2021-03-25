import numpy as np

from src.snc import \
    SteadyStatePolicyAgent
from src.snc.environments import examples


def test_map_state_to_actions():
    np.random.seed(42)
    sim_steps = 1000
    discount_factor = 0.999
    env = examples.double_reentrant_line_only_shared_resources_model(initial_state=np.zeros((4, 1)))
    agent = SteadyStatePolicyAgent(env)
    # TODO test `map_state_to_actions` rather than the simulator.
