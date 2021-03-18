import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sandbox.meio.gsm.convert_excel_to_csv import parse_supply_chain_graphs, write_to_csv, \
    load_into_excel
from meio.experiment.gsm_experiment_utils import run_gsm_with_timeout, plot_gsm
from meio.gsm.utils import GSM, read_supply_chain_from_txt


def get_network(supply_chain):
    sup_chain_file = "../../../meio/supply_chain_dataset/data/..."
    data_xls = load_into_excel(sup_chain_file)
    no_supply = "{:02d}".format(supply_chain)
    stage_configs, up_stages, network_description = parse_supply_chain_graphs(data_xls, no_supply)
    return network_description, stage_configs, up_stages


def get_reported_costs(supply_chain_dir: str):
    with open(os.path.join(supply_chain_dir, "reported_optimal_cost.txt")) as f:
        reported_costs_data = f.readlines()
        reported_costs = {i + 1: int(x.strip()) for i, x in enumerate(reported_costs_data)}
    return reported_costs


def run_gsm(supply_chain: int, timeout: int):
    """
    Run gsm experiment
    :param supply_chain: which supply chain to run on
    :param timeout: interrupt execution after so many seconds
    :return: nothing. Output is stored json file
    """
    """thread worker function"""
    supply_chain_dir = '../../../meio/supply_chain_dataset/data/'
    network_description, stage_configs, up_stages = get_network(supply_chain)
    supply_chain_root = 'supply_chain_dataset/exps/temp{}'.format(supply_chain)
    # generate temporary csv file that gets loaded back in - needed in run_gsm_with_timeout
    write_to_csv(supply_chain_root + '.csv', stage_configs, up_stages)
    print(supply_chain_root + '.csv', network_description)
    # only run relevant chains (no intermediate)
    if network_description['stages_with_broken_down_stoch_lead'] == 0:
        print(supply_chain, 'running gsm')
        results_saved = run_gsm_with_timeout(supply_chain_root+'.csv', GSM.DAG, timeout=timeout)
        with open(supply_chain_root+'.json', 'w') as fp:
            json.dump(results_saved, fp)
        # Use loaded to ensure code below works when loading data (offline analysis of experiments)
        with open(supply_chain_root+'.json', 'r') as fp:
            results = json.load(fp)

        print_supply_chain_log(supply_chain, results, supply_chain_dir, supply_chain_root,
                               network_description, stage_configs, up_stages)
        print(supply_chain, 'finished gsm')

    else:
        print(supply_chain, 'has broken down lead times.')


def print_supply_chain_log(supply_chain: int, results: Optional[Dict], supply_chain_dir: str,
                           supply_chain_root: str, network_description: Dict,
                           stage_configs: Dict, up_stages: Dict, plot: bool = True):
    if results is None:
        print('No results for supply chain {} timeout may have occured'.format(supply_chain))
        return

    reported_stock = pd.read_csv('../../../meio/supply_chain_dataset/data/optimal_inventory_levels.csv')
    ref_stock_row = reported_stock[(reported_stock.SupplyChain == supply_chain) &
                                   (reported_stock.LT == 'mean')]
    reported_safety_stock, reported_base_stock = np.nan, np.nan
    assert ref_stock_row.shape[0] in [0, 1], 'cannot have more than one optimal stock level'
    if ref_stock_row.shape[0] == 1:
        reported_safety_stock = int(ref_stock_row.SafetyStock.values)
        reported_base_stock = int(ref_stock_row.TotalStock.values)

    ref_cost = get_reported_costs(supply_chain_dir)[supply_chain]
    base_stock_levels = np.sum(np.fromiter(results['base_stocks'].values(), dtype=int))
    safety_stock_levels = np.sum(np.fromiter(results['safety_stocks'].values(), dtype=int))
    cost = results['solution']['cost']

    def relative_difference(a, b):
        return np.abs(a - b) / a

    def str_relative_difference(a, b):
        return 'relative difference {:.2f}'.format(relative_difference(a, b))

    print('*' * 30, 'Supply chain', supply_chain, '*' * 30)
    print('Total base stock', base_stock_levels, 'reference=', reported_base_stock,
          str_relative_difference(base_stock_levels, reported_base_stock))
    print('Total safety stock', safety_stock_levels, 'reference=', reported_safety_stock,
          str_relative_difference(safety_stock_levels, reported_safety_stock))
    print('cost={:.2f} reference={:.2f} ratio={:.2f}'.format(cost, ref_cost, ref_cost / cost))
    print('network', network_description, 'stages with base stock', (base_stock_levels > 0).sum())

    if plot:
        plt.ion()
        # generate temporary csv file that gets loaded back in
        # TODO: could avoid writing out intermediate CSV by using convert_to_stages
        write_to_csv(supply_chain_root + '.csv', stage_configs, up_stages)
        print(supply_chain_root + '.csv', network_description)
        stages = read_supply_chain_from_txt(supply_chain_root + '.csv')
        plot_gsm(str(supply_chain), filename=supply_chain_root + '.png',
                 stages=stages,
                 base_stocks=results['base_stocks'], solution=results['solution'], coc_clusters={},
                 do_save=True, show=True)
