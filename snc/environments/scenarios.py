# Disable pylint rule which doesnt like getting the dictionary of parameters from one function then
# passing them as kwargs to another.
# pylint: disable=E1123

from typing import Optional, Dict, Callable
from collections import namedtuple

import numpy as np

from snc.environments import examples
from snc.environments import examples_distribution_with_rebalancing as examples_dwr
from snc.environments.example_product_demo import product_demo_beer_kegs

Scenario = namedtuple('Scenario', ['scenario_name', 'env'])


SCENARIO_CONSTRUCTORS: Dict[str, Callable] = {
    'complex_demand_driven_model': examples.complex_demand_driven_model,
    'complex_demand_driven_model_hot_lots': examples.complex_demand_driven_model,
    'converging_push_model': examples.converging_push_model,
    'dai_wang_model': examples.dai_wang_model,
    'demand_node': examples.demand_node,
    'double_demand_node': examples.double_demand_node,
    'double_reentrant_line_model': examples.double_reentrant_line_model,
    'double_reentrant_line_shared_res_homogeneous_cost':
        examples.double_reentrant_line_only_shared_resources_model,
    'double_reentrant_line_shared_res_different_cost':
        examples.double_reentrant_line_only_shared_resources_model,
    'double_reentrant_line_with_demand_model':
        examples.double_reentrant_line_with_demand_model,
    'double_reentrant_line_with_demand_only_shared_resources_model':
        examples.double_reentrant_line_with_demand_only_shared_resources_model,
    'input_queued_switch_3x3_model': examples.input_queued_switch_3x3_model,
    'klimov_model': examples.klimov_model,
    'ksrs_network_model': examples.ksrs_network_model,
    'multiple_demand_model': examples.multiple_demand_model,
    'one_warehouse': examples_dwr.one_warehouse,
    'product_demo_beer_kegs': product_demo_beer_kegs,
    'push_pull': examples_dwr.push_and_pull_model_example,
    'push_pull_minimal': examples_dwr.push_and_pull_model_minimal_example,
    'simple_link_constrained_model': examples.simple_link_constrained_model,
    'simple_link_constrained_with_route_scheduling_model':
        examples.simple_link_constrained_with_route_scheduling_model,
    'simple_reentrant_line': examples.simple_reentrant_line_model,
    'extended_reentrant_line_model': examples.extended_reentrant_line_model,
    'decoupled_simple_reentrant_line_models': examples.decoupled_simple_reentrant_line_models,
    'single_reentrant_resource':examples.single_reentrant_resource,
    'simple_reentrant_line_model_variance': examples.simple_reentrant_line_model_variance,
    'simple_reentrant_line_homogeneous_cost': examples.simple_reentrant_line_model,
    'simple_routing_model': examples.simple_routing_model,
    'single_server_queue': examples.single_server_queue,
    'single_station_demand_model': examples.single_station_demand_model,
    'tandem_demand_model': examples.tandem_demand_model,
    'three_warehouses_simplified': examples_dwr.three_warehouses_simplified,
    'two_warehouses_simplified': examples_dwr.two_warehouses_simplified,
    'two_warehouses': examples_dwr.two_warehouses_simplified,
    'willems_example_2': examples.willems_example_2
}


def get_scenario_default_params(name: str, job_gen_seed: Optional[int] = None) -> Dict:
    """
    A scenario is an example environment with a specific configuration of hyperparameters.
    This function returns the default hyperparameters based on a name.

    NB: THE SCENARIOS ARE ORDERED ALPHABETICALLY, AND MUST BE IN GLOBAL LIST

    :param name: Scenario name.
    :param job_gen_seed: Seed with which to initialise the environment.
    :return: Dictionary of default argument values for a given environment set up.
    """
    assert name in SCENARIO_CONSTRUCTORS
    if name == 'complex_demand_driven_model':
        return dict(d1=19 / 75, d2=19 / 75, mu1=13 / 15, mu2=26 / 15, mu3=13 / 15, mu4=26 / 15,
                    mu5=1., mu6=2., mu7=1., mu8=2., mu9=1., mu10a=1 / 3, mu10b=1 / 3, mu11=1 / 2,
                    mu12=1 / 10, mud1=100., mus1=100., mud2=100., mus2=100.,
                    cost_per_buffer=np.vstack(
                        (np.ones((12, 1)), np.array([5, 5, 10, 10])[:, None])),
                    initial_state=np.ones((16, 1)) * 1000., capacity=np.ones((16, 1)) * np.inf,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed)

    elif name == 'complex_demand_driven_model_hot_lots':
        return dict(d1=19 / 75, d2=19 / 75, mu1=13 / 15, mu2=26 / 15, mu3=13 / 15, mu4=26 / 15,
                    mu5=1, mu6=2, mu7=1, mu8=2, mu9=1, mu10a=1 / 3, mu10b=1 / 3, mu11=1 / 2,
                    mu12=1 / 10, mud1=100., mus1=100., mud2=0., mus2=100.,
                    cost_per_buffer=np.vstack(
                        (np.ones((12, 1)), np.array([5, 5, 10, 10000])[:, None])),
                    initial_state=np.array(
                        [0, 5, 8, 0, 0, 4, 6, 6, 0, 7, 2, 2, 0, 0, 15, 20])[:, None],
                    capacity=np.ones((16, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'converging_push_model':
        return dict(alpha=1., mu1=1.02, mu2=1.02, mu3=2.08,
                    cost_per_buffer=np.array([1, 1.5, 2])[:, None],
                    initial_state=np.array([0, 0, 200])[:, None],
                    capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'dai_wang_model':
        return dict(alpha1=0.2, mu1=0.66, mu2=0.66, mu3=0.42, mu4=0.42, mu5=0.66,
                    demand_variance=0.2,
                    cost_per_buffer=np.array([[1], [2], [2], [2], [2]]),
                    initial_state=50 * np.ones((5, 1)),
                    capacity=np.ones((5, 1)) * np.inf,
                    job_conservation_flag=False,
                    job_gen_seed=job_gen_seed)

    elif name == 'demand_node':
        return dict(alpha=0.33, mus=0.34, mud=0.9999,
                    cost_per_buffer=np.array([[1],[10]]),
                    initial_state=np.array([[10],[0]]),
                    capacity=np.ones((2, 1)) * np.inf,
                    job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'double_demand_node':
        return dict(alpha1=0.33, alpha2=0.33, mus1=0.68, mus2=0.68, mud1=0.9999, mud2=0.9999,
                    cost_per_buffer=np.array([[1],[2],[10],[20]]),
                    initial_state=np.array([[0],[200],[0],[0]]),
                    capacity=np.ones((4, 1)) * np.inf,
                    job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'double_reentrant_line_model':
        return dict(alpha=1., mu1=4., mu2=3., mu3=2., mu4=3., mu5=4.,
                    cost_per_buffer=np.array([1, 1, 1, 1, 1])[:, None],
                    initial_state=np.array([1, 1, 1, 1, 1])[:, None],
                    capacity=np.ones((5, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'double_reentrant_line_shared_res_homogeneous_cost':
        return dict(alpha=0.33, mu1=0.68, mu2=0.68, mu3=0.68, mu4=0.68,
                    cost_per_buffer=np.ones((4, 1)),
                    initial_state=np.array([1000, 10, 60, 10])[:, None],
                    capacity=np.ones((4, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'double_reentrant_line_shared_res_different_cost':
        return dict(alpha=1., mu1=4., mu2=1.5, mu3=4., mu4=1.5,
                    cost_per_buffer=np.array([1, 10, 1, 10])[:, None],
                    initial_state=np.array([30, 100, 100, 50])[:, None],
                    capacity=np.ones((4, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'double_reentrant_line_with_demand_model':
        return dict(d=1., mu1=4., mu2=3., mu3=2., mu4=3., mu5=4., mus=1e2, mud=1e2,
                    cost_per_buffer=np.array([1, 1, 1, 1, 1, 1, 1])[:, None],
                    initial_state=np.array([1, 1, 1, 1, 1, 1, 1])[:, None],
                    capacity=np.ones((7, 1)) * np.inf,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed)

    elif name == 'double_reentrant_line_with_demand_only_shared_resources_model':
        return dict(d=1., mu1=4., mu2=3., mu3=3., mu4=4., mus=1e2, mud=1e2,
                    cost_per_buffer=np.array([1, 1, 1, 1, 1, 1])[:, None],
                    initial_state=np.array([1, 1, 1, 1, 1, 1])[:, None],
                    capacity=np.ones((6, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'input_queued_switch_3x3_model':
        return dict(mu11=1., mu12=1., mu13=1., mu21=1., mu22=1., mu23=1., mu31=1., mu32=1., mu33=1.,
                    cost_per_buffer=np.ones((9, 1)), initial_state=np.ones((9, 1)) * 10,
                    capacity=np.ones((9, 1)) * np.inf, demand_rate=0.3 * np.ones((9, 1)),
                    job_conservation_flag=True, job_gen_seed=job_gen_seed)

    elif name == 'klimov_model':
        return dict(alpha1=0.2, alpha2=0.3, alpha3=0.4, alpha4=0.5, mu1=1.1, mu2=2.2, mu3=3.3,
                    mu4=4.4, cost_per_buffer=np.ones((4, 1)), initial_state=(0, 0, 0, 0),
                    capacity=np.ones((4, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed, deterministic=False)

    elif name == 'ksrs_network_model':
        return dict(alpha1=8., alpha3=8., mu1=1e3, mu2=10., mu3=1e3, mu4=10.,
                    cost_per_buffer=np.ones((4, 1)), capacity=np.ones((4, 1)) * np.inf,
                    initial_state=np.array([[100], [100], [100], [100]]),
                    job_conservation_flag=True, job_gen_seed=job_gen_seed,
                    list_boundary_constraint_matrices=None, deterministic=False)

    elif name == 'multiple_demand_model':
        return dict(d1=5., d2=5., mu1=12., mu2=9., mu3=12., mu4=9., mus=100., mud=100.,
                    cost_per_buffer=np.array([1, 1, 1, 1, 5, 10, 5, 10])[:, None],
                    initial_state=np.array([0, 0, 0, 0, 10, 0, 10, 0])[:, None],
                    capacity=np.ones((8, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'one_warehouse':
        return dict(d=0.2, mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1,
                    cost_per_buffer=np.ones((3, 1)), initial_state=np.ones((3, 1)),
                    capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'product_demo_beer_kegs':
        return dict(d1=0.006, d2=0.005, d3=0.004,
                    d4=0.006, d5=0.005, d6=0.004,
                    d7=0.006, d8=0.005, d9=0.004,
                    mu1=0.019, mu2=0.019, mu3=0.019,
                    mu4=0.016, mu5=0.016, mu6=0.016,
                    mu7=0.014, mu8=0.014, mu9=0.014,
                    mu10=0.5, mu11=0.5, mu12=0.5,
                    mu13=0.5, mu14=0.5, mu15=0.5,
                    mu16=0.5, mu17=0.5, mu18=0.5,
                    mu19=0.9, mu20=0.9, mu21=0.9,
                    cost_per_buffer=np.ones((21, 1)),
                    initial_state=np.array([[5], [5], [5], [7], [5], [5], [5], [5], [5], [5], [3],
                                            [5], [5], [5], [5], [5], [5], [5], [5], [5], [5]]),
                    capacity=np.ones((21, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'push_pull':
        return dict(d_in=2., d_out=2., mu=3., mud=10.,
                    initial_state=np.array([[100], [100], [100]]), cost_per_buffer=np.ones((3, 1)),
                    capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'push_pull_minimal':
        return dict(d_in=2., d_out=2., mu=3., job_gen_seed=job_gen_seed,
                    initial_state=np.array([[150], [100]]), capacity=np.ones((2, 1)) * np.inf,
                    job_conservation_flag=True)

    elif name == 'simple_link_constrained_model':
        return dict(alpha1=4., mu12=2., mu13=10., mu25=10., mu32=5., mu34=20., mu35=20., mu45=10.,
                    mu5=100., cost_per_buffer=np.ones((5, 1)),
                    initial_state=np.vstack((100 * np.ones((1, 1)), np.zeros((4, 1)))),
                    capacity=np.ones((5, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'simple_link_constrained_with_route_scheduling_model':
        return dict(alpha1=2.9, mu12=2, mu13=10, mu25=1, mu32=5, mu34=2, mu35=2, mu45=10, mu5=20,
                    cost_per_buffer=np.ones((5, 1)),
                    initial_state=np.vstack((40 * np.ones((1, 1)), np.zeros((4, 1)))),
                    capacity=np.ones((5, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'simple_reentrant_line':
        return dict(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68,
                    initial_state=np.array([1000, 10, 10])[:, None],
                    cost_per_buffer=np.array([1.5, 1, 2])[:, None],
                    capacity=np.ones((3, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed, deterministic=False)

    elif name == 'extended_reentrant_line_model':
        return dict(alpha1=0.33, mu1=0.69, mu2=0.35, mu3=0.68, mu4=0.35,
                    initial_state=np.array([1000, 0, 0, 1000, 1000, 0])[:, None],
                    cost_per_buffer=np.array([1, 2, 3, 2.5, 2, 5])[:, None],
                    capacity=np.ones((6, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed, deterministic=False)

    elif name == 'decoupled_simple_reentrant_line_models':
        return dict(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68, mu4=0.99,
                    initial_state=np.array([1000, 10, 100, 1000, 10, 100, 10])[:, None],
                    cost_per_buffer=np.array([1.5, 1, 2, 1.5, 1, 2, 5])[:, None],
                    capacity=np.ones((7, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed, deterministic=False)

    elif name == 'single_reentrant_resource':
        return dict(alpha1=0.33, mu1=0.68001, mu2=0.68,
                    initial_state=np.array([1000, 0])[:, None],
                    cost_per_buffer=np.array([1, 1])[:, None],
                    capacity=np.ones((2, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'simple_reentrant_line_homogeneous_cost':  # Validated
        return dict(alpha1=9., mu1=22., mu2=10., mu3=22.,
                    cost_per_buffer=np.ones((3, 1)),
                    initial_state=np.array([100, 100, 100])[:, None],
                    capacity=np.ones((3, 1)) * np.inf,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed,
                    deterministic=False)

    elif name == 'simple_reentrant_line_with_demand_model':
        return dict(alpha_d=2., mu1=5., mu2=2.5, mu3=5., mus=1e3, mud=1e3,
                    cost_per_buffer=np.ones((5, 1)),
                    initial_state=np.array([10, 25, 55, 0, 100])[:, None],
                    capacity=np.ones((5, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed)

    elif name == 'simple_reentrant_line_model_variance':
        return dict(alpha1=0.33, mu1=0.68, mu2=0.35, mu3=0.68,
                    demand_variance=0.33,
                    cost_per_buffer=np.ones((3, 1)),
                    initial_state=np.array([[0], [0], [0]]),
                    job_gen_seed=job_gen_seed)

    elif name == 'simple_routing_model':
        return dict(alpha_r=19., mu1=13., mu2=7., mu_r=20.,
                    cost_per_buffer=np.array([[1.], [1.], [2.]]),
                    initial_state=np.array([[100.], [100.], [100.]]),
                    capacity=np.ones((3, 1)) * np.inf,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed)

    elif name == 'single_server_queue':
        return dict(cost_per_buffer=np.ones((1, 1)), initial_state=np.zeros((1, 1)),
                    capacity=np.ones((1, 1)) * np.inf, demand_rate_val=0.7,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed,
                    deterministic=False)

    elif name == 'single_station_demand_model':
        return dict(alpha_d=9., mu=10., mus=90., mud=50.,
                    cost_per_buffer=np.array([5, 1, 10])[:, None],
                    initial_state=np.array(([0, 0, 10])), capacity=np.ones((3, 1)) * np.inf,
                    job_conservation_flag=True, job_gen_seed=job_gen_seed)

    elif name == 'tandem_demand_model':
        return dict(n=3, d=2, mu=np.array([4, 5, 6, 7, 8]),
                    cost_per_buffer=np.array([4, 5, 6, 7, 8])[:, None],
                    initial_state=np.array([6, 6, 6, 6, 6]),
                    capacity=np.ones((5, 1)) * np.inf,
                    job_conservation_flag=True)

    elif name == 'three_warehouses_simplified':
        return dict(d1=2.5, d2=2.5, d3=2.5, mu1=10., mu2=100., mu3=2.5, mu5=0.1,
                    mu6=0., mu7=10., mu8=100, mu9=3.8, mu11=5., mu12=0.,
                    mu13=10., mu14=100., mu15=2.5, mu17=0.1, mu18=0., mu19=0.1,
                    mu20=5., mu21=0.1,
                    cost_per_buffer=np.array([[1], [1], [5], [1], [1], [5], [1], [1], [5]]),
                    initial_state=np.array([0, 0, 100, 0, 0, 0, 0, 0, 100]),
                    job_conservation_flag=True, job_gen_seed=job_gen_seed,
                    r_to_w_rebalance=False)

    elif name == 'two_warehouses_simplified':
        return dict(d1=5., d2=5., mu1=6., mu2=100., mu3=10., mu5=3., mu6=0., mu7=10.,
                    mu8=100., mu9=5., mu11=3., mu12=0.,
                    cost_per_buffer=np.array([[1], [1], [10], [1], [1], [10]]),
                    initial_state=np.array([0, 0, 0, 0, 0, 100])[:, None],
                    job_conservation_flag=True, job_gen_seed=job_gen_seed,
                    r_to_w_rebalance=False, w_to_w_rebalance=True)

    elif name == 'two_warehouses':
        return dict(d1=0.05, d2=0.05, mu1=0.1, mu2=0.1, mu3=0.1, mu4=0.1, mu5=0.5, mu6=0.1,
                    mu7=0.1, mu8=0.1, mu9=0.1, mu10=0.1, mu11=0.1, mu12=0.1,
                    cost_per_buffer=np.ones((6, 1)), initial_state=np.ones((6, 1)),
                    capacity=np.ones((6, 1)) * np.inf, job_conservation_flag=True,
                    job_gen_seed=job_gen_seed, w_to_w_rebalance=True, r_to_w_rebalance=False)

    elif name == 'willems_example_2':
        return dict(initial_state=np.array(
            [10, 9, 8, 7, 6, 10, 9, 8, 7, 6, 10, 9, 8, 7, 6, 10, 9])[:, None],
                    capacity=np.ones((17, 1)) * np.inf, job_conservation_flag=True,
                    single_activity_per_resource=False, job_gen_seed=job_gen_seed,
                    resource_capacity=np.array([560000, 70000, 60000, 320000, 40000, 30000,
                                                1040000, 130000, 120000, 88000, 11000, 10000,
                                                10000, 300000, 100000, 400000, 40000, 250000]))
    else:
        raise Exception('No Scenario Named: {}'.format(name))


def load_scenario(name: str, job_gen_seed: Optional[int],
                  override_env_params: Optional[Dict] = None) -> Scenario:
    """
    Takes the environment parameters as input, and used them to customise a particular CRW example.

    NB: THE SCENARIOS ARE ORDERED ALPHABETICALLY, AND MUST BE IN GLOBAL LIST

    :param name: Scenario name.
    :param job_gen_seed: Seed with which to initialise the environment.
    :param override_env_params: Dictionary of environment parameters to override defaults.
    :return: Scenario(name, env):
        - name: Scenario name.
        - env: CRW environment with the parameters defined by this scenario.
    """
    assert name in SCENARIO_CONSTRUCTORS, f'No Scenario Named: {name}.'
    if override_env_params is not None:
        override_env_params['job_gen_seed'] = job_gen_seed
    else:
        override_env_params = dict()

    env_params = get_scenario_default_params(name, job_gen_seed)
    env_params.update(override_env_params)
    return Scenario(name, SCENARIO_CONSTRUCTORS[name](**env_params))
