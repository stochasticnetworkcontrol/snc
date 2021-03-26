# Plots cross-correlations between buffer states and arrivals in the simple link-constrained network.
# This facilitates a visualization of the distributed impact of arrivals throughout the network.
# Network corresponds with Figure 6.7 in CTCN.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snc.simulation.validation_script import process_parsed_args, run_validation
from argparse import Namespace

snc_dir = os.path.join(os.getenv('HOME'), 'snc')
corr_dir = os.path.join(snc_dir, 'experiment', 'correlations')
env_dir = os.path.join(snc_dir, 'snc', 'simulation', 'json_examples_validation_script', 'env')
agent_dir = os.path.join(snc_dir, 'snc', 'simulation', 'json_examples_validation_script', 'agent')

args = Namespace(agents='bs_hedgehog',
                 art_run=False,
                 debug_info=False,
                 discount_factor=0.99999,
                 env_name='simple_link_constrained_model',
                 env_param_overrides=os.path.join(env_dir, 'simple_link_routing.json'),
                 hedgehog_param_overrides=os.path.join(agent_dir, 'hedgehog.json'),
                 logdir=corr_dir,
                 maxweight_param_overrides='{}',
                 num_steps=10000,
                 rl_agent_params=None,
                 rl_checkpoints=None,
                 seed=0,
                 server_mode=True)
args = process_parsed_args(args)


def run_correlational_analysis(data_fname):
    data = pd.read_json(data_fname)

    # get buffer state time series
    state = np.array(data.state.values.tolist())
    n_step, n_buffer = state.shape

    # get arrivals time series
    arrivals = np.array(data.arrivals.values.tolist())
    arrivals_buffer1 = arrivals[:,0]

    fig, axes = plt.subplots(1, 1, figsize=(8,5))

    # plot cross-correlations
    ax = axes
    for buffer in range(n_buffer):
        ax.xcorr(arrivals_buffer1, state[:,buffer], normed=False, maxlags=None, usevlines=False, linestyle='-', marker='', label=f'Buffer {buffer+1}')
    ax.set_title('Cross-correlations between buffer 1 arrivals and buffer queue lengths')
    ax.set_xlabel(r'Time interval $h$')
    ax.set_ylabel(r'$Q(t+h)\cdot \alpha_1(t)$')
    ax.legend()
    ax.set_xlim([0,n_step])
    ax.set_ylim([0,None])
    plt.tight_layout()
    plt.savefig('correlational_analysis.png')
    plt.show()


if __name__ == "__main__":
    save_locations = run_validation(args)
    data_fname = os.path.join(save_locations['bs_hedgehog'], 'datadict.json')
    run_correlational_analysis(data_fname)
