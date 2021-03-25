import argparse
import os
import re
from logging import warning

import meio.gsm.tree_gsm as tree_gsm
from meio.experiment.gsm_experiment_utils import plot_gsm
from meio.gsm.dag_gsm import GuaranteedServiceModelDAG
from meio.gsm.utils import read_supply_chain_from_txt


def run_gsm(path, network, figpath, run_gsm_optimiser, plotting=True):
    # Load data
    supply_chain_filename = os.path.join(path, network)
    stages = read_supply_chain_from_txt(supply_chain_filename)
    # Run GSM
    gsm = GuaranteedServiceModelDAG(stages)

    if not run_gsm_optimiser:
        solution = None
        base_stocks = None
    else:
        solution = gsm.find_optimal_solution()
        base_stocks = tree_gsm.compute_base_stocks(solution.policy, stages)

    if plotting:
        plot_gsm(args.network, filename=os.path.join(figpath, network), stages=stages,
                 base_stocks=base_stocks, solution=solution.serialize(),
                 do_save=True)
    return gsm


if __name__ == "__main__":
    WHITELIST = [r'bulldozer.txt', r'food.txt', r'toys.txt', r'semiconductors.txt', r'willems_.*']

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="directory to load data from",
                        default='../../tests/meio/gsm/')
    parser.add_argument("--network", help="network csv file", default='bulldozer.txt')
    parser.add_argument("--figpath", help="directory to save figures to", default='./figs')
    parser.add_argument("--no-solution", action='store_true',
                        help="do not find a solution for the graph", default=False)
    parser.add_argument("--plot", action='store_true', help="generate network plots", default=True)

    args = parser.parse_args()

    if not any(re.search(r, args.network) for r in WHITELIST):
        warning('Unexpected network {}'.format(args.network))
    run_gsm(args.path, args.network, args.figpath, run_gsm_optimiser=~args.no_solution,
            plotting=args.plot)
