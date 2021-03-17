import argparse
import json
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sandbox.meio.gsm.analysis_utils import run_gsm, get_network, print_supply_chain_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Running gsm")
    parser.add_argument('--plot', action='store_true', help='dont run anything, just plot')
    parser.add_argument('--timeout', type=int, default=60*60*72, const=None)
    parser.add_argument('--chain', type=int, default=None, const=None, help='run this chain')
    assert parser.parse_args().timeout > 0
    if parser.parse_args().chain is not None:
        chains = [parser.parse_args().chain]
    else:
        chains = [i for i in range(1, 39)]
    if not parser.parse_args().plot:
        print('running jobs')
        # run jobs
        jobs = []
        for i in chains:
            p = multiprocessing.Process(target=run_gsm, args=(i, parser.parse_args().timeout))
            jobs.append(p)
            p.start()

    # post processing results
    print('plotting jobs')
    results = dict()
    results["metadata"] = {}
    results["data"] = {}
    results["metadata"]["timeout"] = parser.parse_args().timeout
    network_descriptions = {}
    willems_dir = '{}/../../../meio/willems_dataset/data/'
    for i in chains:
        supply_chain_results = 'willems_dataset/exps/temp{}'.format(i)
        if os.path.isfile(supply_chain_results+'.json'):
            with open(supply_chain_results+'.json', 'r') as fp:
                results_single = json.load(fp)
                if results_single is not None:
                    results["data"][i] = results_single
                    print('supply chain {} loaded.'.format(i))
                    network_descriptions[i], stage_configs, up_stages = get_network(i)
                    print_supply_chain_log(i, results["data"][i], willems_dir, supply_chain_results,
                                           network_descriptions[i], stage_configs, up_stages,
                                           plot=False)
        else:
            print('No experimental result file found ', supply_chain_results+'.json')

    if len(results["data"]) == 0:
        print('no results found - terminating')
        sys.exit(0)

    def getfig():
        fig = plt.figure()
        ax = fig.subplots(1)
        ax.set_ylabel("execution time (mins)")
        ax.grid()
        return fig, ax


    def get_execution_times(network_feature):
        return [(s, value["execution_time"], network_descriptions[s][network_feature]) for s, value in results["data"].items()]

    a = pd.DataFrame(get_execution_times('total_stages'),
                     columns=('supply chain', 'execution time', 'total stages'))
    print('Execution times\n', a.sort_values(by='execution time'))
    # plot results
    plt.ion()
    _, x, total_stages = zip(*get_execution_times('total_stages'))
    fig, ax = getfig()
    ax.set_xlabel("number of stages")
    ax.scatter(total_stages, np.array(x)/60, label='DAG', s=100)
    ax.set_title('Execution times {} chains'.format(len(x)))
    fig.savefig('execution_times_total_stages.png')

    _, x, total_stages = zip(*get_execution_times('stages_demand'))
    fig, ax = getfig()
    ax.set_xlabel("number of demand stages")
    ax.scatter(total_stages, np.array(x) / 60, label='DAG', s=100)
    ax.set_title('Execution times {} chains'.format(len(x)))
    fig.savefig('execution_times_stages_demand.png')
