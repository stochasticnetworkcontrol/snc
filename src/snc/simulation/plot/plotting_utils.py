from contextlib import contextmanager

import matplotlib
import numpy as np
from typing import Optional, Dict, List, Callable
import matplotlib.pyplot as plt

from matplotlib import axes
from matplotlib import animation
from matplotlib import lines as mlines

import snc.utils.snc_types as types
import snc.simulation.plot.drawing_utils as drawing_utils


def plot_results(cost_log: types.Array1D, label: str, state_log: types.StateProcess,
                 time_interval: float = 1, plot_cost=True, ymax: Optional[float] = -np.inf):
    plt.ioff()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.set_xlabel("Simulation steps")
    ax.set_ylabel("Cost")
    x_ticks = np.arange(0, cost_log[0].size *
                        time_interval, step=time_interval)
    if plot_cost:
        ax.plot(x_ticks, cost_log[0], label=label)
    for i, s in enumerate(state_log, 1):
        ax.plot(x_ticks, s, label='lane ' + str(i))
    if ymax > - np.inf:
        plt.ylim([0, ymax])
    elif plot_cost:
        plt.ylim([0, np.max((np.max(cost_log), np.max(state_log)))])
    else:
        plt.ylim([0, np.max(state_log)])
    ax.legend()
    plt.show()


def plot_results_mean_percentiles(
        data_log: Dict[str, np.ndarray], time_interval: float = 1, label: str = 'buffer ',
        plot_cost=True, y_max: Optional[float] = - np.inf):
    # data_log = {'cost': np.ndarray,  [simulation_steps, num_simulations],
    #            'state': np.ndarray,  [simulation_steps, state.size, num_simulations]}

    cost_mean = np.mean(data_log['cost'], axis=1)
    cost_25q = np.quantile(data_log['cost'], 0.25, axis=1)
    cost_75q = np.quantile(data_log['cost'], 0.75, axis=1)

    state_mean = np.array(np.mean(data_log['state'], axis=2))
    state_25q = np.quantile(data_log['state'], 0.25, axis=2)
    state_75q = np.quantile(data_log['state'], 0.75, axis=2)

    fig = plt.figure()
    ax = fig.subplots(1)
    ax.grid()
    ax.set_xlabel("Simulation steps")
    if plot_cost:
        ax.set_ylabel("Buffer sizes and Total cost")
    else:
        ax.set_ylabel("Buffer sizes")
    x_ticks = np.arange(0, cost_mean.size * time_interval, step=time_interval)

    alpha = 0.3

    # Plot mean
    if plot_cost:
        ax.plot(x_ticks, cost_mean, label='cost')

    for i, s in enumerate(state_mean.T, 1):
        ax.plot(x_ticks, s, label='lane ' + str(i))

    # Plot quantiles
    if plot_cost:
        ax.fill_between(x_ticks, cost_25q, cost_75q,
                        alpha=alpha, edgecolor=None)
    for i, (s_25, s_75) in enumerate(zip(state_25q.T, state_75q.T), 1):
        ax.fill_between(x_ticks, s_25, s_75, alpha=alpha, edgecolor=None)

    # Set y_lim
    if y_max > - np.inf:
        plt.ylim([0, y_max])
    elif not plot_cost:
        plt.ylim([0, np.max(state_75q)])
    else:
        plt.ylim([0, np.max((cost_75q, state_75q))])

    # Legend for state variables
    if plot_cost:
        display = range(0, state_mean.shape[1] + 1)
    else:
        display = range(0, state_mean.shape[1])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handle for i, handle in enumerate(handles) if i in display],
              [label for i, label in enumerate(labels) if i in display])

    plt.show()


def tsplot(x_ticks: np.ndarray, y_perc_min: np.ndarray, y_perc_max: np.ndarray, ax):
    alpha = 0.3
    ax.fill_between(x_ticks, y_perc_min, y_perc_max,
                    alpha=alpha, edgecolor=None)
    return plt.gca()


#################
# ANIMATION PLOTS
#################


def init_cost_axes(ax: axes.Axes, cost_log: types.Array1D, state_log: types.StateProcess,
                   ymax: float, time_interval: float, plot_cost: bool) -> None:
    """ Initialise the axes (labels and limits) for a cost plot

    :param ax: the axes to plot on
    :param cost_log: the total cost at various times  [time_steps]
    :param state_log: the state at various times  [state x time_steps]
    :param ymax: max y value on the plot
    :param time_interval: the time interval of the simulation (for axes scaling)
    :param plot_cost: boolean of whether to plot the total cost
    """
    ax.grid()
    ax.set_xlabel("Simulation steps")
    if plot_cost:
        ax.set_ylabel("Cost")
    else:
        ax.set_ylabel("Buffer State")
    if ymax > - np.inf:
        ax.set_ylim(*[0, ymax])
    elif plot_cost:
        ax.set_ylim(*[0, np.max((np.max(cost_log), np.max(state_log)))])
    else:
        ax.set_ylim(*[0, np.max(state_log)])
    ax.set_xlim(*[0, cost_log.shape[1] * time_interval])


def init_draw_plot(ax: axes.Axes, width: float, height: float) -> None:
    """Initialise the axes for a drawing plot

    :param ax: the axes to plot on
    :param width: the width of the plot (x axis)
    :param height: the height of the plot (y axis)
    """
    padding = 4
    ax.set_xlim(0, width)
    ax.set_ylim(-padding, height + padding)

    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)


def init_cost_lines(ax: axes.Axes, plot_cost: bool, colors: Dict) -> List[mlines.Line2D]:
    """ Initialise the lines for the plot of cost on some axes

    :param ax: the axes to plot on
    :param plot_cost: a boolean whether to plot the cost
    :param colors:  the colors dictionary
    """
    lines = []
    if plot_cost:
        line_cost, = ax.plot([], [], label='(step) cost',
                             color=colors['neutral'])
        lines.append(line_cost)
    else:
        for i, c in enumerate(colors['buffers'], 1):
            line, = ax.plot([], [], label='lane ' + str(i), color=c)
            lines.append(line)
    return lines


def plot_dual_schematic_cost_animations(data_dicts: List[Dict], draw_fn: Callable, colors: Dict,
                                        time_interval: float = 1.0, plot_cost: bool = True,
                                        ymax: float = -np.inf, pbar=None,
                                        do_annotations: bool = True,
                                        width: float = 20.0, height: float = 8.0,
                                        scaling_factor: Optional[float] = 1.0,
                                        seconds: int = 10,
                                        fps: int = 80) \
        -> animation.FuncAnimation:
    """ For each data_dict in data_dicts, provide a subplot with an animation in the
    left half and a the cost and buffer state in the right half.

    :param data_dicts: List of data dictionaries, standard result format from simulator
    :param draw_fn: A function which will plot a network diagram. This function takes the args:
        (ax, state, action, demand, effects, max_capacity, colors, time_step, cost, do_annotations)
        and returns None. A reference can be seen in `drawing_utils.draw_reentrant_line`
    :param colors: a dictionary of colors to draw the buffers, demand and neutral
    :param time_interval: the time interval of the simulation (to scale axes)
    :param plot_cost: whether to plot the cost
    :param ymax: the maximum y value on the line plot
    :param pbar: a progress bar to update during the animation (tqdm instance)
    :param do_annotations: whether to annotate the drawings with number of items in buffers
        demand, and actions
    :param width: basic width of the plot, it will be multiplied by the scaling factor
    :param height: basic height of the plot, it will be multiplied by the scaling factor
    :param scaling_factor: parameter that specify the scaling factor for the graph animation
    :param seconds: length of the video
    :param fps: frame per second
    """
    num_data_dicts = len(data_dicts)
    my_dpi = 96

    fig = plt.figure(figsize=(12, 6 * num_data_dicts), dpi=my_dpi)

    inits = []
    animates = []
    for i, data in enumerate(data_dicts, 1):
        ax1 = fig.add_subplot('{}2{}'.format(
            num_data_dicts, 2 * i - 1), frameon=False)
        ax2 = fig.add_subplot('{}2{}'.format(num_data_dicts, 2 * i))
        init, animate = _add_dual_schematic_cost_animation(ax1, ax2, data, draw_fn, colors,
                                                           time_interval, plot_cost, ymax,
                                                           do_annotations, scaling_factor,
                                                           width, height)
        inits.append(init)
        animates.append(animate)

    data_points = len(data_dicts[0]['action'])
    frames = seconds * fps  # 80fps
    frames_index = np.linspace(0, data_points - 1, num=frames, dtype=int)

    def all_init():
        ret = []
        for i in inits:
            ret.extend(i())
        return ret

    def all_animate(i):
        if pbar is not None:
            pbar.update(1)
        ret = []
        for a in animates:
            ret.extend(a(frames_index[i]))
        return ret

    anim = animation.FuncAnimation(fig, all_animate, init_func=all_init,
                                   frames=frames, interval=(1000 / fps), blit=True)
    return anim


def _add_dual_schematic_cost_animation(ax1: axes.Axes, ax2: axes.Axes, data: Dict,
                                       draw_fn: Callable, colors: Dict, time_interval: float,
                                       plot_cost: bool, ymax: float, do_annotations: bool,
                                       scaling_factor: float, width: float, height: float):
    """ Add a subplot of a drawn diagram and cost plot animation to each axis,
    returning an init and update function for use in matplotlib.animation.FuncAnimation

    :param ax1: matplotlib.axes.Axes object on which to draw our schematic animation
    :param ax2: matplotlib.axes.Axes object on which to plot our cost animation
    :param draw_fn: A function which will plot a network diagram. This function takes the args:
        (ax, state, action, demand, effects, max_capacity, colors, time_step, cost, do_annotations)
        and returns None. A reference can be seen in `drawing_utils.draw_reentrant_line`
    :param colors: a dictionary of colors to draw the buffers, demand and neutral
    :param time_interval: the time interval of the simulation (to scale axes)
    :param plot_cost: whether to plot the cost
    :param ymax: the maximum y value on the line plot
    :param do_annotations: whether to annotate the drawings with number of items in buffers
        demand, and actions
    :param scaling_factor: parameter that specify the scaling factor for the graph animation
    :param width: basic width of the plot, it will be multiplied by the scaling factor
    :param height: basic height of the plot, it will be multiplied by the scaling factor
    """

    cost_log = np.asarray(data['cost']).T
    state_log = np.asarray(data['state']).T
    demand_log = np.asarray(data['arrivals']).T
    action_log = np.asarray(data['action']).T
    processing_log = np.asarray(data['processing']).T

    width *= scaling_factor
    height *= scaling_factor
    init_draw_plot(ax1, width, height)

    init_cost_axes(ax2, state_log, cost_log, ymax, time_interval, plot_cost)
    lines = init_cost_lines(ax2, plot_cost, colors)

    def init():
        ax2.legend()
        for l in lines:
            l.set_data([], [])
        return lines

    x_ticks = np.arange(cost_log[0].size * time_interval, step=time_interval)
    cost = [0]

    def animate(i):
        x = x_ticks[:i]

        if plot_cost:
            y = cost_log[0][:i]
            lines[-1].set_data(x, y)
            if y.shape[0] > 0:
                cost[0] += (0.999 ** i) * y[-1]
        else:
            for j, s in enumerate(state_log):
                lines[j].set_data(x, s[:i])

        drawing_utils.draw_reset(ax1, width, height)
        max_capacity = np.max(state_log[:, 0]) * 1.2
        draw_fn(ax1, state_log[:, i], action_log[:, i], demand_log[:, i], processing_log[:, i],
                max_capacity, colors, i, cost[0], do_annotations)
        return lines

    return init, animate


# main in order to test only how the *static* plot looks like
if __name__ == '__main__':
    FIG = plt.figure(figsize=(8, 6))
    AX = FIG.add_subplot(111)
    init_draw_plot(AX, 27, 20)

    state = np.ones(9) * 50
    action = np.ones(15)
    effects = np.ones(15) * 50
    COLORS = {
        'buffers': [
            '#bc5d86', '#ac3267', '#501e36',
            '#91dbbe', '#70b39c', '#1c433c',
            '#70a9e2', '#7f8fd2', '#8f74b8'
        ],
        'suppliers': ['#cc87a5', '#befcda', '#6ec1ea'],
        'neutral': '#000000'
    }

    drawing_utils.draw_three_warehouses_simplified(AX, state, action, None, effects,
                                                   max_capacity=100,
                                                   colors=COLORS, time_step=5,
                                                   cumul_cost=5000, do_annotations=True)

    plt.show()


@contextmanager
def non_gui_matplotlib():
    """This is a context manager to turn off the GUI backend of matplotlib for testing"""
    backend = matplotlib.get_backend()
    matplotlib.use('agg')
    yield
    matplotlib.use(backend)
