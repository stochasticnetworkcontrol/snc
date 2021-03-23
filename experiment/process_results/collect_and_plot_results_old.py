from typing import List, Dict, Optional

from os import listdir
from os.path import join, isdir
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt


def read_experiment_runs(experiment_path: str) -> Dict[str, List]:
    """
    Function loops through the seed instances (runs) of conducted validation experiment
    and collects together cost histories for each agent

    :param experiment_path: name of the directory where the different seeds for a single
    scenario experiments are stored
    """
    exp_runs = defaultdict(list)  # type: Dict[str, List]
    seed_instances = listdir(experiment_path)
    for seed_instance in seed_instances:
        agent_instances = listdir(join(experiment_path, seed_instance))
        for agent_instance in agent_instances:
            if not isdir(join(experiment_path, seed_instance, agent_instance)):
                continue

            filename = join(experiment_path, seed_instance, agent_instance, "cost.json")
            try:
                with open(filename) as f:
                    cost_data = json.load(f)
            except:
                print(f"failed to read: {filename}")
                continue

            exp_runs[agent_instance].append((np.array(cost_data).ravel()))
    return exp_runs


def aggregate_exp_runs(
        exp_runs: Dict[str, List],
        cumsum: bool = False,
        discount_factor: float = 1.,
        population_variance: bool = False,
        save_fig: Optional[str] = None
):
    """
    Function plots the average cost (instantaneous or cumulative)
    and confidence intervals of this average trends
    or expected deviation from the average trajectory

    :param exp_runs: dictionary storing the cost histories for multiple seed runs indexed by agent
        name.
    :param cumsum: Whether to consider cumulative or instantaneous cost.
    :param discount_factor: discount factor in cumulative cost objective.
    :param population_variance: whether to plot population spread as error bars instead of mean
        estimation error.
    :param save_fig: pathname of the image file if to be saved on disk instead of displaying.
    """

    f, ax = plt.subplots(1, 1, figsize=(12, 6))
    # colors = ["r","b","g","k","m","y"] # TODO introduce color pallete for different agents

    for _, agent in enumerate(exp_runs):
        # to compute per timestep statistics we need to truncate the histories to the same
        # duration
        min_dur = min(len(a) for a in exp_runs[agent])
        agent_exp_runs_list = [a[:min_dur] for a in exp_runs[agent]]
        agent_exp_runs = np.vstack(agent_exp_runs_list)
        if cumsum:
            log_discount_multipliers = np.arange(min_dur) * np.log(discount_factor)
            agent_exp_runs = agent_exp_runs * np.exp(log_discount_multipliers)
            agent_exp_runs = np.cumsum(agent_exp_runs, axis=1)
            y_label = "Cumulative cost"
        else:
            y_label = "Instantaneous cost"

        n = len(agent_exp_runs)

        mean_cost = np.mean(agent_exp_runs, axis=0)
        st_cost = np.std(agent_exp_runs, axis=0, ddof=1)
        mean_estimation_error = st_cost / np.sqrt(n)

        if population_variance:
            # either we plot expected variation in individual trajectories
            ax.fill_between(np.arange(len(np.array(mean_cost))),
                            mean_cost - st_cost,
                            mean_cost + st_cost, alpha=0.2)
        else:
            # or we plot estimation error of the mean trajectory
            ax.fill_between(np.arange(len(np.array(mean_cost))),
                            mean_cost + mean_estimation_error,
                            mean_cost - mean_estimation_error, alpha=0.2)

        ax.plot(mean_cost, label=agent)
        ax.set_ylabel(y_label)
        ax.set_xlabel("Timestep")
        ax.legend()

    if save_fig is not None:
        f.savefig(save_fig)
    else:
        plt.show()


# TODO. THIS FILE HAS BEEN DEPRECATED.
#  You can find same functionality in "benchmark_cumulative_cost.py".
if __name__ == "__main__":
    experimental_path = ""  # select the path
    e_runs = read_experiment_runs(experimental_path)

    aggregate_exp_runs(e_runs, cumsum=True, discount_factor=1.,
                       save_fig="test.png")
