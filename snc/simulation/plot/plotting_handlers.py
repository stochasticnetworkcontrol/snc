import itertools
import numpy as np
from matplotlib import axes
from typing import List, Iterable, Optional, Tuple

import snc.simulation.utils.validation_utils as validation_utils
from snc.simulation.plot.base_handlers import LinePlotHandler
import snc.utils.snc_types as types

from snc.environments.job_generators.scaled_bernoulli_services_poisson_arrivals_generator import (
    ScaledBernoulliServicesPoissonArrivalsGenerator
)
from snc.environments.job_generators.scaled_bernoulli_services_and_arrivals_generator import (
    ScaledBernoulliServicesAndArrivalsGenerator
)


class StateCostPlotter(LinePlotHandler):
    """
    Live plot the evolution of buffer levels and total cost per step.
    """

    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.001,
                 testing_mode: bool = False, do_plot_cost: bool = True,
                 do_plot_state: bool = True, **kwargs):
        self.do_plot_cost = do_plot_cost
        self.do_plot_state = do_plot_state
        super().__init__(
            num_simulation_steps, time_interval, plot_frequency, ax, pause, testing_mode, **kwargs)

    @property
    def line_labels(self) -> Iterable[str]:
        labels = []
        if self.do_plot_cost:
            labels.append('Instantaneous Cost')
        if self.do_plot_state:
            buffers = self.num_lines - len(labels)
            labels.extend(['Buffer {}'.format(i) for i in range(1, buffers + 1)])
        return labels

    @property
    def y_label(self) -> str:
        return 'Cost' if self.do_plot_cost else 'Buffer State'

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):

        start, end = self.get_start_end(step)

        lines = []
        if self.do_plot_cost:
            lines.append(np.array(data_dict['cost'][start:end]).T)
        if self.do_plot_state:
            lines.append(np.array(data_dict['state'][start:end]).T)

        return np.vstack(lines)


class WorkloadPlotter(LinePlotHandler):
    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.001,
                 testing_mode: bool = False, plot_fluid_model: bool = False,
                 plot_hedging: bool = False, **kwargs):
        """
        Live plot the evolution of workload (w) and its projection on the monotone region (w_star)
        for along each dimension of workload space. It can also plot the evolution of the fluid
        model, and the evolution of the hedging threshold, if the flags are activated.

        :param num_simulation_steps: Number of simulation steps to plot.
        :param time_interval: Time interval over which the job generator draws events.
        :param plot_frequency:  How often we update the plot.
        :param ax: Matplotlib axes to plot on (if None, create new figure).
        :param pause: Time interval to pause and update the matplotlib GUI in plotting.
        :param testing_mode: Flag that controls whether to suppress the GUI for testing.
        :param plot_fluid_model: Flag that controls whether to plot the fluid model in the same
            axes as the workload.
        :param plot_hedging: Flag that controls whether to plot the hedging threshold level in the
            same axes as the workload.
        """
        self.plot_fluid_model = plot_fluid_model
        if self.plot_fluid_model:
            self.idling_cumsum_cache = np.array([])
        self.plot_hedging = plot_hedging
        super().__init__(num_simulation_steps, time_interval, plot_frequency, ax, pause,
                         testing_mode, **kwargs)
        self.num_simulated_blocks = 0

    def init_lines_and_cache(self) -> None:
        """Initialise the lines on the plot and draw the y label.
        """
        super().init_lines_and_cache()
        if self.plot_fluid_model:
            self.idling_cumsum_cache = np.full((self.num_workload_lines, self.num_simulation_steps),
                                               np.nan)

    @property
    def num_workload_lines(self):
        _num_workload_lines = self.num_lines
        if self.plot_hedging:
            # Subtract the line corresponding to beta_star.
            _num_workload_lines = _num_workload_lines - 1
        if self.plot_fluid_model:
            # Divide by 3 because we plot each w, w_star, and w_fluid.
            _num_workload_lines = _num_workload_lines // 3
        else:
            # Divide by 2 because we plot each w and w_star.
            _num_workload_lines = _num_workload_lines // 2
        return _num_workload_lines

    @property
    def line_styles(self) -> Iterable[Tuple[str, float]]:
        styles = []
        if self.plot_hedging:
            # beta_star is plotted as markers so there is no line ('')
            styles.append(('', 1))
        for _ in range(1, self.num_workload_lines + 1):
            if self.plot_fluid_model:
                # w_star is plotted as a dotted line (':'), w and w_fluid_model are plotted as a
                # solid line ('-')
                styles.extend([('-', 1), (':', 2), ('-', 1)])
            else:
                # w_star is plotted as a dotted line (':') and w is plotted as a solid line ('-')
                styles.extend([('-', 1), (':', 2)])
        return styles

    @property
    def line_markers(self) -> List[Tuple[str, int]]:
        markers = super().line_markers
        if markers and self.plot_hedging:
            markers[0] = ('_', 2)
        return markers

    @property
    def line_labels(self) -> Iterable[str]:
        labels = []
        if self.plot_hedging:
            labels.append('beta star')
        for i in range(1, self.num_workload_lines + 1):
            if self.plot_fluid_model:
                labels.extend(['W {}'.format(i), 'W* {}'.format(i), 'W {} fluid'.format(i)])
            else:
                labels.extend(['W {}'.format(i), 'W* {}'.format(i)])
        return labels

    @property
    def y_label(self) -> str:
        label = 'Workload process'
        if self.plot_fluid_model:
            label += ', Fluid process'
        if self.plot_hedging:
            label += ', beta_star'
        return label

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):
        # @TODO: This function is far too long, a refactor would be great.
        start, end = self.get_start_end(step)
        num_data_points = end - start

        if not self.reproduce_mode:
            assert len(reporter_cache['agent']) == 1
            assert len(reporter_cache['agent'][0].workload_tuple) == 3
            workload_mat = reporter_cache['agent'][0].workload_tuple.workload_mat
        else:
            workload_mat = np.array(reporter_cache['agent']["workload_tuple"][1])

        num_workload_vec = workload_mat.shape[0]

        w_process = workload_mat @ np.array(data_dict['state'][start:end]).T
        w_star_process = np.full((num_workload_vec, num_data_points), np.nan)

        lines = []

        if self.plot_hedging:
            beta_star_process = np.ones((1, num_data_points)) * np.nan

        cum_num_mpc_steps = np.cumsum(reporter_cache['num_mpc_steps'])
        j = int(np.searchsorted(cum_num_mpc_steps, start))
        # the condition below ensures that cum_num_mpc_steps is
        # not indexed outside of its range
        i_thres = cum_num_mpc_steps[j] if len(cum_num_mpc_steps) >= j+1 else 1
        for i in range(num_data_points):
            # switch to the next set of network state configurations
            # if it has been computed
            if start + i > i_thres and len(cum_num_mpc_steps) >= j+2:
                j += 1
                i_thres = cum_num_mpc_steps[j]

            if not self.reproduce_mode:
                w_star = reporter_cache['strategic_idling_tuple'][j].w_star
            else:
                w_star = np.array(reporter_cache['strategic_idling_tuple']["w_star"][j])
            if w_star is not None:
                w_star_process[:, i] = w_star.reshape(-1)

            if self.plot_hedging:
                if not self.reproduce_mode:
                    beta_star = reporter_cache['strategic_idling_tuple'][j].beta_star
                else:
                    beta_star = reporter_cache['strategic_idling_tuple']["beta_star"][j]
                if beta_star is not None:
                    beta_star_process[:, i] = beta_star
        if self.plot_hedging:
            lines.append(beta_star_process)

        if self.plot_fluid_model:
            actions = np.array(data_dict['action'][start:end]).T
            if end > 1:
                if not self.reproduce_mode:
                    buffer_processing_matrix = \
                        reporter_cache['env'][0].job_generator.buffer_processing_matrix
                else:
                    buffer_processing_matrix = \
                        np.array(reporter_cache["env"]["_job_generator"] \
                                     ["_buffer_processing_matrix"])

                idling_start_end = validation_utils.compute_idling_process_from_workload(
                    workload_mat, buffer_processing_matrix, actions)
                idling = np.hstack((np.zeros((num_workload_vec, 1)), idling_start_end))
            else:
                idling = np.zeros((num_workload_vec, 1))

            if start > 0:
                idling_cumsum = np.tile(
                    self.idling_cumsum_cache[:, [start - 1]], (1, num_data_points)) + \
                                np.cumsum(idling, axis=1)
            else:
                idling_cumsum = np.cumsum(idling, axis=1)

            if self.num_lines > 0:
                self.idling_cumsum_cache[:, start:end] = idling_cumsum

            if not self.reproduce_mode:
                demand_rate = reporter_cache['env'][0].job_generator.demand_rate.T
            else:
                demand_rate = np.array(reporter_cache["env"]["_job_generator"]["_demand_rate"]).T

            rho = demand_rate
            delta_mat = np.tile(1. - rho, (1, num_data_points))
            w_0 = workload_mat @ np.array(data_dict['state'][0]).reshape(-1, 1)

            total_time_mat = np.tile(np.arange(start, end).reshape(1, -1),
                                     (num_workload_vec, 1))
            w_fluid_process = np.tile(w_0.reshape(-1, 1), (1, num_data_points)) \
                              - np.multiply(delta_mat, total_time_mat) + idling_cumsum

            is_scaled_job_gen = isinstance(reporter_cache['env'][0].job_generator,
                                           (ScaledBernoulliServicesPoissonArrivalsGenerator,
                                            ScaledBernoulliServicesAndArrivalsGenerator))
            if not is_scaled_job_gen:
                total_time_mat *= self.time_interval
                w_fluid_process *= self.time_interval

            for w_i, w_star_i, w_fluid_i in zip(w_process, w_star_process, w_fluid_process):
                lines.extend([w_i, w_star_i, w_fluid_i])
        else:
            for w_i, w_star_i in zip(w_process, w_star_process):
                lines.extend([w_i, w_star_i])

        return np.vstack(lines)


class HedgingThresholdPlotter(LinePlotHandler):
    """
    Live plot the evolution of the hedging threshold (beta_star).
    """

    @property
    def line_labels(self) -> Iterable[str]:
        return ['beta_star', 'h']

    @property
    def y_label(self) -> str:
        return 'Hedging'

    @property
    def line_styles(self) -> Iterable[Tuple[str, float]]:
        return [('', 1), ('-', 1)]

    @property
    def line_markers(self) -> List[Tuple[str, int]]:
        return [('_', 2), ('', 2)]

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):
        start, end = self.get_start_end(step)
        num_data_points = end - start

        beta_star_process = np.ones((1, num_data_points)) * np.nan
        height_process = np.ones((1, num_data_points)) * np.nan

        cum_num_mpc_steps = np.cumsum(reporter_cache['num_mpc_steps'])
        j = int(np.searchsorted(cum_num_mpc_steps, start))
        # the condition below ensures that cum_num_mpc_steps is
        # not indexed outside of its range
        i_thres = cum_num_mpc_steps[j] if len(cum_num_mpc_steps) >= j+1 else 1
        for i in range(num_data_points):
            # switch to the next set of network state configurations
            # if it has been computed
            if start + i > i_thres and len(cum_num_mpc_steps) >= j+2:
                j += 1
                i_thres = cum_num_mpc_steps[j]

            if not self.reproduce_mode:
                beta_star = reporter_cache['strategic_idling_tuple'][j].beta_star
                h = reporter_cache['strategic_idling_tuple'][j].height_process
            else:
                beta_star = reporter_cache['strategic_idling_tuple']["beta_star"][j]
                if "height_process" in reporter_cache['strategic_idling_tuple']:
                    h = reporter_cache['strategic_idling_tuple']["height_process"][j]
                else:
                    h = None

            assert beta_star is not None

            beta_star_process[:, i] = beta_star
            height_process[:, i] = h

        return np.vstack((beta_star_process, height_process))


class IdlingPlotter(LinePlotHandler):
    """
    Plot the live evolution of the cumulative idling at each station up to current time t:
    sum_{j=0}^t (1 - C @ u(j)).
    """

    @property
    def line_labels(self) -> Iterable[str]:
        labels = [('Station {}'.format(i + 1) for i in itertools.count())]
        return itertools.chain.from_iterable(labels)

    @property
    def y_label(self) -> str:
        return 'Idling'

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):
        start, end = self.get_start_end(step)
        if not self.reproduce_mode:
            constituency_matrix = reporter_cache['env'][0].constituency_matrix
        else:
            constituency_matrix = np.array(reporter_cache['env']["_constituency_matrix"])
        actions = np.array(data_dict['action'][start:end]).T
        idling = np.ones((constituency_matrix.shape[0], actions.shape[1])) - \
                 constituency_matrix @ actions
        if start > 0:
            idling_cumsum = np.tile(self.data_cache[:, [start - 1]], (1, actions.shape[1])) + \
                            np.cumsum(idling, axis=1)
        else:
            idling_cumsum = np.cumsum(idling, axis=1)

        return idling_cumsum


class EffectiveCostErrorPlotter(LinePlotHandler):
    """ Plot the live change in the states and the effective state of the simulation
    """

    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.001,
                 testing_mode: bool = False, eff_cost_err_method: str = 'absolute', **kwargs):
        """

        :param num_simulation_steps: Number of simulation steps to plot.
        :param time_interval: Time interval over which the job processor draws events.
        :param plot_frequency: how often to update the plots with all the points.
        :param ax: the matplotlib axes to plot on (if None, create new figure).
        :param pause: Time interval to pause and update the matplotlib GUI in plotting.
        :param testing_mode: Whether to suppress the GUI for testing.
        :param eff_cost_err_method: method to compute the effective cost error. Accepted values are:
            'relative', 'absolute', or 'invariant-to-load'.
        """

        assert eff_cost_err_method in {'relative', 'absolute', 'invariant-to-load'}
        self.eff_cost_err_method = eff_cost_err_method
        super().__init__(num_simulation_steps, time_interval, plot_frequency, ax, pause,
                         testing_mode, **kwargs)

    @property
    def line_labels(self) -> List[str]:
        return ['Effective Cost Error']

    @property
    def y_label(self) -> str:
        return 'Effective Cost Error'

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):

        start, end = self.get_start_end(step)
        num_data_points = end - start

        if not self.reproduce_mode:
            workload_mat = reporter_cache['agent'][0].workload_tuple.workload_mat
        else:
            workload_mat = np.array(reporter_cache['agent']["workload_tuple"][1])

        buff_process = np.array(data_dict['state'][start:end]).T
        w_process = workload_mat @ buff_process

        c_bars = np.zeros(w_process.shape)
        assert num_data_points == c_bars.shape[1]
        cum_num_mpc_steps = np.cumsum(reporter_cache['num_mpc_steps'])
        j = int(np.searchsorted(cum_num_mpc_steps, start))
        # the condition below ensures that cum_num_mpc_steps is
        # not indexed outside of its range
        i_thres = cum_num_mpc_steps[j] if len(cum_num_mpc_steps) >= j+1 else 1
        for i in range(num_data_points):
            # switch to the next set of network state configurations
            # if it has been computed
            if start + i > i_thres and len(cum_num_mpc_steps) >= j+2:
                j += 1
                i_thres = cum_num_mpc_steps[j]

            if not self.reproduce_mode:
                c_bar = reporter_cache['strategic_idling_tuple'][j].c_bar
            else:
                c_bar = np.array(reporter_cache['strategic_idling_tuple']["c_bar"][j])

            if c_bar is None:
                c_bars[:, i] = c_bars[:, i - 1]
            else:
                c_bars[:, i] = c_bar.reshape(-1)

        effective_cost = np.einsum('ij, ij-> j', c_bars, w_process)
        buff_cost = np.array(data_dict['cost'][start:end]).T

        if self.eff_cost_err_method == 'relative':
            eff_cost_error = (buff_cost - effective_cost) / buff_cost
        elif self.eff_cost_err_method == 'absolute':
            eff_cost_error = buff_cost - effective_cost
        elif self.eff_cost_err_method == 'invariant-to-load':
            eff_cost_error = (buff_cost - effective_cost) / (1 + np.log(1 + effective_cost))
        else:
            raise Exception('Effective error method is not valid')

        lines = eff_cost_error
        return lines


class ArrivalsPlotter(LinePlotHandler):
    """
    Plot the cumulative actual demand/arrivals MINUS the cumulative expected demand/arrivals at each
    time step for all buffers, i.e., plot the random walk of the zero-mean noise in
    the arrival process.
    """

    @property
    def line_labels(self) -> Iterable[str]:
        labels = [('Buffer {}'.format(i + 1) for i in itertools.count())]
        return itertools.chain.from_iterable(labels)

    @property
    def y_label(self) -> str:
        return 'Arrivals'

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int):
        start, end = self.get_start_end(step)

        if start > 0:
            last_arrivals = self.data_cache[:, max(start - 1, 0)]
        else:
            last_arrivals = 0
        num_steps = end - start
        if not self.reproduce_mode:
            demand_rate = reporter_cache['env'][0].job_generator.demand_rate.T
        else:
            demand_rate = np.array(reporter_cache["env"]["_job_generator"]["_demand_rate"]).T

        arrivals = np.array(data_dict['arrivals'][start:end])
        expected_arrivals = np.ones((num_steps, demand_rate.shape[0])) * demand_rate

        residual = arrivals - expected_arrivals
        cumulative_arrivals = last_arrivals + np.cumsum(residual, axis=0)
        return cumulative_arrivals.T


class CumulativeCostPlotter(LinePlotHandler):
    """ Plot the live change in the buffer state of the simulation, and the total
    step cost.
    """

    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.1,
                 testing_mode: bool = False, discounted: bool = True,
                 **kwargs):
        """

        :param num_simulation_steps: Number of simulation steps to plot.
        :param time_interval: Time interval over which the job processor draws events.
        :param plot_frequency: how often to update the plots with all the points.
        :param ax: the matplotlib axes to plot on (if None, create new figure).
        :param pause: Time interval to pause and update the matplotlib GUI in plotting.
        :param testing_mode: Whether to suppress the GUI for testing.
        :param discounted: specify whether the cumulative cost is discounted or not.
        """
        self.discounted = discounted
        super().__init__(
            num_simulation_steps, time_interval, plot_frequency, ax, pause, testing_mode,
            **kwargs
        )

    @property
    def line_labels(self) -> Iterable[str]:
        labels = [['Total Cost']]
        return itertools.chain.from_iterable(labels)

    @property
    def y_label(self) -> str:
        if self.discounted:
            return 'Cumulative Disc Cost'
        else:
            return 'Cumulative Cost'

    def runtime_check(self, data_dict: types.DataDict, step: int, gamma: float) -> None:
        """
        This method perform a one-off check that cumulative cost
        has been accumulated correctly by recomputing the current sum
        value from the beginning of simulation.

        :param data_dict: dictionary that contains information (among which the costs) about the
        steps done so far.
        :param step: step of the simulation.
        :param gamma: discount factor used in the cumulative cost
        """
        start, _ = self.get_start_end(step)

        data = np.array(data_dict['cost'][:start+1]).T
        if self.discounted:
            factors = np.power(gamma, self.time_interval * np.arange(start+1))
            data *= factors
        current_cum_sum = np.sum(data, axis=1)
        assert self.data_cache[:,start] == current_cum_sum,\
            (self.data_cache[:,:start],current_cum_sum,data,factors)

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int) -> np.ndarray:
        """
        Function that computes the cumulative costs for the iterations done since the last plot
        update. The cumulative cost can be either discounted or not.

        :param reporter_cache: it is used for logging and contains the discount factor.
        :param data_dict: dictionary that contains information (among which the costs) about the
        steps done so far.
        :param step: step of the simulation.
        :return ndarray containing the cumulative costs for the last X iterations,
        where X is defined by the trigger_frequency.
        """
        if not self.reproduce_mode:
            gamma = reporter_cache['discount_factor'][0]
        else:
            gamma = reporter_cache['discount_factor']

        if step == 3 * self.trigger_frequency:
            self.runtime_check(data_dict, step, gamma)

        start, end = self.get_start_end(step)

        if step > 0:
            new_step_cost = np.array(data_dict['cost'][start+1:end]).T
            if self.discounted:
                new_step_cost = new_step_cost * \
                    np.power(gamma, self.time_interval * np.arange(start+1,end))

            new_cum_step_cost = np.zeros((new_step_cost.shape[0],new_step_cost.shape[1]+1))
            new_cum_step_cost[:,1:] = np.cumsum(new_step_cost, axis=1)
            last_step_cost = self.data_cache[:,[max(start, 0)]]
        else:
            new_step_cost = np.array(data_dict['cost'][0:1]).T
            new_cum_step_cost = new_step_cost
            last_step_cost = 0

        output = last_step_cost + new_cum_step_cost

        return output


class ActionsToFluidPlotter(LinePlotHandler):
    """
    This plotter tracks the ratio of cumulative number of times each activity has been performed
    over the cumulative activity rates given by fluid policy at each timestep.
    """
    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.001,
                 testing_mode: bool = False):
        """
        :param num_simulation_steps: Number of simulation steps to plot.
        :param time_interval: Time interval over which the job processor draws events.
        :param plot_frequency: how often to update the plots with all the points.
        :param ax: the matplotlib axes to plot on (if None, create new figure).
        :param pause: Time interval to pause and update the matplotlib GUI in plotting.
        :param testing_mode: Whether to suppress the GUI for testing.
        """
        super().__init__(
            num_simulation_steps, time_interval, plot_frequency, ax, pause, testing_mode
        )
        self._activity_rates: Optional[np.ndarray] = None
        self._arrival_rate: Optional[np.ndarray] = None

    @property
    def line_labels(self) -> Iterable[str]:
        labels: List[str] = []
        entities = self.num_lines
        labels.extend(['Activity {}'.format(i) for i in range(1, entities + 1)])
        return labels

    @property
    def y_label(self) -> str:
        return 'Actual actions to fluid rates cumulative ratio'

    def get_new_line_data(self, reporter_cache: types.ReporterCache,
                          data_dict: types.DataDict, step: int) -> np.ndarray:
        """
        :param reporter_cache: it is used for logging and contains the discount factor.
        :param data_dict: dictionary that contains information (among with the costs) about the
                          steps done so far.
        :param step: step of the simulation.
        """
        start, end = self.get_start_end(step)

        lines = []

        step_actions = np.array(data_dict['action'][:end]).T
        cum_actions = np.cumsum(step_actions, axis=1)[:, start:]

        step_fluid_policy = np.array(data_dict['zeta_star'][:end]).T
        cum_fluid_policy = np.cumsum(step_fluid_policy, axis=1)[:, start:]

        output = cum_actions/cum_fluid_policy

        lines.append(output)

        return np.vstack(lines)
