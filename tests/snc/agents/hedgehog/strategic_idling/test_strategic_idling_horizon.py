import numpy as np

from src.snc import StrategicIdlingParams
from src.snc import \
    StrategicIdlingCoreHorizon, StrategicIdlingGTOHorizon
from src import snc as wl
import src.snc.environments.examples as examples


def test_zero_workloads_input():
    cost_per_buffer = np.array([1.5, 1, 2])[:, None]
    env = examples.simple_reentrant_line_model(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68,
                                               cost_per_buffer=cost_per_buffer)
    num_wl_vec = 2
    load, workload_mat, _ = wl.compute_load_workload_matrix(env, num_wl_vec)
    strategic_idling_params = StrategicIdlingParams()
    horizon = 100

    si_object = StrategicIdlingCoreHorizon(workload_mat,
                                           load,
                                           cost_per_buffer,
                                           env.model_type,
                                           horizon,
                                           strategic_idling_params)

    si_gto_object = StrategicIdlingGTOHorizon(workload_mat,
                                              load,
                                              cost_per_buffer,
                                              env.model_type,
                                              horizon,
                                              strategic_idling_params)

    x_zero = np.array([[0],[0],[0]])
    si_output = si_object.get_allowed_idling_directions(x_zero)
    assert np.all(si_output.w_star >= 0)

    si_output = si_gto_object.get_allowed_idling_directions(x_zero)
    assert np.all(si_output.w_star >= 0)
