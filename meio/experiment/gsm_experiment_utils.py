import os
import json
import signal
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from graphviz import Digraph
from scipy.stats.distributions import poisson

from meio.gsm.tree_gsm import GSM_Solution, InconsistentGSMConfiguration, \
    IncompatibleGraphTopology, GSMException, UnSupportedGSMException, \
    compute_expected_inventories, compute_base_stocks
from meio.gsm.utils import create_gsm_instance, GSM


# TODO: write test


def plot_gsm(network: str, filename: str, stages: Dict, base_stocks: Dict, solution: Dict,
             do_save: bool, save_format: str = 'png', show=True):
    # For examples see https://graphviz.readthedocs.io/en/stable/examples.html
    # on Ubuntu run "sudo apt install python-pydot python-pydot-ng graphviz"
    g = Digraph(network, format=save_format, filename=filename)

    def get_stage_shape_description(stage_name):
        node_color = 'grey'
        if base_stocks:
            base_stock_level = "B=" + str(base_stocks[stage_name])
            if base_stocks[stage_name] > 0:
                node_color = 'blue'
        else:
            base_stock_level = ''
        si = "SI=" + str(solution['policy'][stage_name]['si']) if solution else ''
        stage_description = "{}\n{}\n{}".format(stage_name, base_stock_level, si)
        shape = 'diamond' if stages[stage_name].is_ext_demand_stage else 'square'

        return dict(label=stage_description, shape=shape, color=node_color)

    # add all stages.
    # We could avoid adding nodes that are already in a cluster since they would have been added
    # at the previous step. However duplication doesn't hurt the visualisation and performance
    # is not an issue so let's keep it simple
    for stage_name, stage_struct in stages.items():
        g.node(stage_name, **get_stage_shape_description(stage_name))
    # add edges
    for stage_name, stage_struct in stages.items():
        edge_color = 'grey'
        S = str(solution['policy'][stage_name]['s']) if solution else ''
        for down_name, _ in stage_struct.down_stages.items():
            g.edge(stage_name, down_name, label=S, color=edge_color)
        # TODO: Can we assert up_stage is complementary?
    if do_save:
        g.render(filename)
        # g.save(filename=filename+'.dot')
    if show:
        g.view()

    # TODO: For CoC show identified bipartite graph
    # TODO: demand bound, k? stages['Final_Assembly'].demand_thres
    # Ei max demand  stages['Final_Assembly'].max_s_time
    # TODO: pooling strategy
    # max draining time: stages['Final_Assembly'].max_s_bound (every node)


def simulate_gsm(demand_mean=3, lead_time=12, threshold=0.95, N=200000):
    """
    This simulates the "inventory position" at a single node/service station or a single stage,
    under a Possion demand profile for each time step.

    It is an attempt to simulate, as precisely as possible and consequently, as abstractly as
    necessarily, a policy generated in accordance with Willem's published GSM framework.

    It models the "inventory position" for a given demand and lead time for when we expect a given
    "failure rate".

    Lead time: Time to collect any necessary materials from where ever they are process an SKU by
               the node/warehouse/service station.

    Failure rate is described at one point in the original Willem's paper as "the percentage of time
    that the safety stock covers the demand variation".  This is an over-arching explanation, but to
    be more precise, it is the percentage of time steps for which the "inventory position" is
    negative.

    N.B.  This is not the same as the percentage of SKUs that were not available when required.

    Inventory position: Calculated per time step, after any deliveries have been recieved, it is
    the number of SKUs in stock - the number of SKUs that are outstanding because they were not
    available when required.

    We assume that SKUs that were not available when required are ordered on the time step that
    were found to be unavailable and these unfullfilled orders are fullfilled in exactly the time
    in takes to replenish an SKU.

    The time to replenish an SKU (tau [Willems]) is lead time because we assume the service time of
    the node is zero and it has no upstream supply nodes, so incoming service time is zero.

    As part of the simulation. the base stock levels required at t=0 for the desired failure rate
    to be achieved, according to Willem's algorithm, are calculated for this simple, single station
    "network".

    :param demand_mean: The mean number of SKUs demanded per timestep.
    :param lead_time: The time it takes to replenish an SKU.
    :param threshold: The percentage of days for which the "inventory position" is negative.
    :param N: The number of time steps in the simulation
    """

    p = poisson(demand_mean * lead_time)
    # Base stock in a integer, so we define bounds:
    max_base_stock = p.ppf(threshold)
    min_base_stock = max_base_stock - 1

    inventory_position = np.array([])

    s = np.random.poisson(size=N, lam=demand_mean)  # Sample demand for all time steps
    s = np.append([0] * lead_time, s)  # Add a buffer of no demand to simulate the time nothing has
    # had a chance to be replenished.

    # Compute the inventory position at "the end of" each (simulated-instantaneous) time step:
    for i in range(N):
        inventory_position = np.append(inventory_position,
                                       min_base_stock - np.sum(s[i:i + lead_time]))

    # Calculate the percentage of "failed" days:
    lowest_expected_failure_rate = np.mean(inventory_position < -(max_base_stock - min_base_stock))
    highest_expected_failure_rate = np.mean(inventory_position < 0)

    print("With a base stock of: {}, the simulation has an error rate of {}".format(
        min_base_stock,
        highest_expected_failure_rate))

    print("With a base stock of: {}, the simulation has an error rate of {}".format(
        max_base_stock,
        lowest_expected_failure_rate))

    failure_rate = (1 - threshold)
    print("The desired failure rate of {} IS{}within the simulated bounds".format(
        round(failure_rate, 3),
        " " if (lowest_expected_failure_rate < failure_rate < highest_expected_failure_rate)
        else " NOT "))

    return inventory_position


def get_willems_filename(data_set_id):
    return "willems_{:02d}.csv".format(data_set_id)


class GSMTimeOutException(GSMException):
    """
    For timing out running a GSM algorithm on a large/complex dataset
    """
    pass


def handler(signum, frame):
    """
    For timing out running a GSM algorithm on a large/complex dataset.
    Part of the signal module.
    """
    raise GSMTimeOutException("Time out")


def run_gsm_with_timeout(data_set_filename: str, gsm_type: GSM, timeout: int = 15) \
        -> Optional[Dict]:
    """
    Run the specified version of gsm
    """
    execution_start_time = datetime.utcnow()
    _, gsm = create_gsm_instance(gsm_type, data_set_filename)
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    results = None
    try:
        solution = gsm.find_optimal_solution()  # type: GSM_Solution
        signal.alarm(0)  # Cancel timeout
        execution_time = (datetime.utcnow() - execution_start_time).total_seconds()

        safety_stocks = compute_expected_inventories(solution.policy, gsm.stages)
        base_stocks = compute_base_stocks(solution.policy, gsm.stages)
        results = dict(execution_time=execution_time, solution_cost=solution.cost,
                       solution=solution.serialize(), safety_stocks=safety_stocks,
                       base_stocks=base_stocks)
    except GSMException:
        print("The {} version of GSM timed out for {}".format(gsm_type,
                                                              data_set_filename))

    return results


def run_gsm_type_on_all_willems_datasets(dir_name: str, gsm_type: GSM,
                                         results: Dict,
                                         inventory_holding_rates: Dict[int, float],
                                         data_set_ids: List[int] = None,
                                         timeout: int = 15,
                                         json_filename: str = 'results.json') -> None:
    """
    Run the specified version of gsm of all the data sets (by default, although a subset can be
    specified).  Accumulate and print the results.

    :param dir_name: The directory within which the Willems data set config files are stored.
    :param results: Where to store the results of the experiments (cost and execution time).
    :param gsm_type: The type of the GSM model (e.g. spanning tree or clusters of commonality).
    :param inventory_holding_rates:  To modify cost.  If not specified for a data set id (key),
                                     assumed to be 1.
    :param data_set_ids:  A potentially restricted set of data set identifiers [1..38]
    :param timeout:  A number of seconds for ending a run of an algorithm on a dataset if taking
                     too long.
    :param json_filename: filename to save results dictionary. This is updated incrementally for
        each dataset
    """

    assert all(list(map(lambda x: 0 < x < 39, data_set_ids)))  # Check data set ids are valid
    assert results["metadata"]["timeout"] == timeout, 'cannot mix results with diff timeouts'

    for data_set_id in data_set_ids:
        data_set_filename = get_willems_filename(data_set_id)
        try:
            execution_start_time = datetime.utcnow()

            _, gsm = create_gsm_instance(gsm_type, os.path.join(dir_name, data_set_filename))

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)
            try:
                # TODO: could use run_gsm_with_timeout
                sol = gsm.find_optimal_solution()  # type: GSM_Solution

                signal.alarm(0)  # Cancel timeout

                sol_s = sol.serialize()

                solution_cost = sol_s["cost"] * inventory_holding_rates.get(data_set_id, 1)

                execution_time = (datetime.utcnow() - execution_start_time).total_seconds()

                print("In {:6} seconds, the {} version of GSM computes a "
                      "total cost of {} for {}".format(round(execution_time, 2),
                                                       gsm_type,
                                                       solution_cost,
                                                       data_set_filename))

                results["data"].setdefault(data_set_id, {})["{}".format(gsm_type)] = \
                    {"execution_time": execution_time,
                     "solution_cost": solution_cost}

                if json_filename is not None:
                    with open(json_filename, 'w') as fp:
                        json.dump(results, fp)

            except GSMTimeOutException:
                print("The {} version of GSM timed out for {}".format(gsm_type,
                                                                      data_set_filename))

        except UnSupportedGSMException:
            print("Skipping all files as model type not supported".format())
            break
        except IncompatibleGraphTopology:
            print("Skipping {} as not a compatible topology".format(data_set_filename))
            continue
        except InconsistentGSMConfiguration:
            print("Skipping {} as network topology labels are not as expected.".format(
                data_set_filename))
            continue
