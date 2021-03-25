from abc import ABC, abstractmethod
from typing import List, Optional, Iterable, Tuple

from src import snc as types

from matplotlib import pyplot as plt
from matplotlib import axes
from tqdm import tqdm
import matplotlib
import numpy as np


class Handler(ABC):
    """A class to handle report generation, set off by a Reporter object.

    A Handler is a callable base class that gets called by Reporter at each time step.
    These are designed to handle useful events/reports like logging or live plotting.
    """

    def __init__(self, trigger_frequency: int):
        """:param trigger_frequency: How often the handler actually performs its action."""
        self.trigger_frequency = trigger_frequency

    def __call__(self, reporter_cache: types.ReporterCache, data_dict: types.DataDict, step: int):
        """Generate a report, using the current state of the data dictionary, the
        reporters cache, and current time step. This method is called every time step.

        :param reporter_cache: A dictionary of items in the reporters cache.
        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        if step % self.trigger_frequency == 0:
            return self.handle(reporter_cache, data_dict, step)

    @abstractmethod
    def handle(self, reporter_cache: types.ReporterCache, data_dict: types.DataDict, step: int):
        """Generate a report, using the current state of the data dictionary, the
        reporters cache, and current time step. This method is called every time step.

        :param reporter_cache: A dictionary of items in the reporters cache.
        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        pass


class PrintStateHedging(Handler):
    def handle(self, reporter_cache: types.ReporterCache,
               data_dict: types.DataDict, step: int) -> None:
        start = max(step - self.trigger_frequency, 0)
        end = step + 1
        num_data_points = end - start
        if reporter_cache['num_mpc_steps']:
            for i in range(num_data_points):
                j = np.searchsorted(np.cumsum(reporter_cache['num_mpc_steps']), start + i)
                beta_star = reporter_cache['strategic_idling_tuple'][j].beta_star
                if beta_star is None:
                    beta_star = 'None'
                print('State: ', data_dict['state'][start + i],
                      ' | Action: ', data_dict['action'][start + i],
                      ' | beta_star: ', beta_star)


class PrintStateHeuristic(Handler):
    def handle(self, reporter_cache: types.ReporterCache,
               data_dict: types.DataDict, step: int) -> None:
        start = max(step - self.trigger_frequency, 0)
        end = step + 1
        num_data_points = end - start
        for i in range(num_data_points):
            print('State: ', data_dict['state'][start + i],
                  ' | Action: ', data_dict['action'][start + i])


class PrintActualtoFluidRatio(Handler):
    """
    This print hadnler tracks the ratio of cumulative number of times each activity has been
    performed over the cumulative activity rates given by fluid policy at each timestep.
    """
    def __init__(self, trigger_frequency: int):
        super().__init__(trigger_frequency)
        self.total_sum_actions: Optional[np.ndarray] = None
        self.total_fluid_sum_actions: Optional[np.ndarray] = None

    def handle(self, reporter_cache: types.ReporterCache,
               data_dict: types.DataDict, step: int) -> None:
        start = max(step - self.trigger_frequency, 0)
        end = step + 1

        if step > 0:
            new_actions = np.array(data_dict['action'][start+1:end]).T
            self.total_sum_actions += np.sum(new_actions, axis=1, keepdims=True)

            new_fluid_actions = np.array(data_dict['zeta_star'][start+1:end]).T
            self.total_fluid_sum_actions += np.sum(new_fluid_actions, axis=1, keepdims=True)

        else:
            self.total_sum_actions = np.array(data_dict['action'][0:1]).T
            self.total_fluid_sum_actions = np.array(data_dict['zeta_star'][0:1]).T

        print("Actual to fluid ratio: ",
              (self.total_sum_actions /
               self.total_fluid_sum_actions).flatten(), "\n")


class PrintInverseLoadings(Handler):
    """
    This handler tracks ratio of how many jobs in expectation
    each activity has processed to the total amount of expected arrival jobs that
    has accumulated since the begining of the simulation. Inverse of this quantity is equivalent
    to the load of the activity.
    """
    def __init__(self, trigger_frequency: int):
        super().__init__(trigger_frequency)
        self.total_sum_actions: Optional[np.ndarray] = None
        self.total_fluid_sum_actions: Optional[np.ndarray] = None
        self._processing_rates: Optional[np.ndarray] = None
        self._arrival_rate: Optional[np.ndarray] = None

    def handle(self, reporter_cache: types.ReporterCache,
               data_dict: types.DataDict, step: int) -> None:
        start = max(step - self.trigger_frequency, 0)
        end = step + 1

        # Only networks with a single source of arrivals can be processed correctly for now.
        if np.sum(reporter_cache['env'][0].job_generator.demand_rate > 0) != 1:
            return
        if self._processing_rates is None:
            drain_matrix = reporter_cache['env'][0].job_generator.draining_jobs_rate_matrix
            assert np.all(np.sum(drain_matrix < 0, axis=1) == 1), \
                "This printer handler is not valid for routing networks."
            self._processing_rates = -np.sum(drain_matrix, axis=0)[:, None]
            self._arrival_rate = np.sum(reporter_cache['env'][0].job_generator.demand_rate)

        if step > 0:
            new_actions = np.array(data_dict['action'][start+1:end]).T
            self.total_sum_actions += np.sum(new_actions, axis=1, keepdims=True)

            new_fluid_actions = np.array(data_dict['zeta_star'][start+1:end]).T
            self.total_fluid_sum_actions += np.sum(new_fluid_actions, axis=1, keepdims=True)

        else:
            self.total_sum_actions = np.array(data_dict['action'][0:1]).T
            self.total_fluid_sum_actions = np.array(data_dict['zeta_star'][0:1]).T

        actions_inverse_loading = self.total_sum_actions * self._processing_rates / \
            (self._arrival_rate * step)
        fluid_inverse_loading = self.total_fluid_sum_actions * self._processing_rates / \
            (self._arrival_rate * step)

        print("Actual actions inverse loading: ", actions_inverse_loading.ravel())
        print("Fluid actions inverse loading: ", fluid_inverse_loading.ravel())


class ProgressBarHandler(Handler):
    """ A Handler object that prints a progress bar, over the course of the simulation."""

    def __init__(self, num_simulation_steps: int, trigger_frequency: Optional[int] = 1):
        """
        :param num_simulation_steps: The number of simulation steps to run.
        :param trigger_frequency: How often the handler actually performs the reporting update.
        """

        self.num_simulation_steps = num_simulation_steps
        self.progress_bar = tqdm(total=self.num_simulation_steps)
        super().__init__(trigger_frequency)

    def handle(self, reporter_cache: types.ReporterCache, data_dict: types.DataDict, step: int):
        """Generate a report, using the current state of the data dictionary, the
        reporters cache, and current time step. This method is called every time step.

        :param reporter_cache: A dictionary of items in the reporters cache.
        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        if step == 0:
            num_update = 1
        else:
            num_update = self.trigger_frequency
        self.progress_bar.update(num_update)


class LinePlotHandler(Handler):
    """ A Handler object that creates and live updates a line chart, over the course
    of the simulation.
    """

    def __init__(self, num_simulation_steps: int, time_interval: float,
                 plot_frequency: int, ax: Optional[axes.Axes] = None, pause: float = 0.001,
                 testing_mode: bool = False, reproduce_mode: bool = False):
        """
        :param num_simulation_steps: Number of simulation steps to plot.
        :param time_interval: Time interval over which the job processor draws events.
        :param plot_frequency: how often to update the plots with all the points.
        :param ax: the matplotlib axes to plot on (if None, create new figure).
        :param pause: Time interval to pause and update the matplotlib GUI in plotting.
        :param testing_mode: Whether to suppress the GUI for testing.
        :param reproduce_mode: Whether the handler is used to reproduce the simulation run.
        """

        self.num_simulation_steps = num_simulation_steps
        self.time_interval = time_interval
        self.pause = pause
        self.do_testing_mode = testing_mode
        self.reproduce_mode = reproduce_mode

        if ax is None:
            self.fig = plt.figure(figsize=(6, 6))
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = ax

        if not self.do_testing_mode:
            plt.ion()  # make interactive matplotlib
        self.ax.grid()
        self.ax.set_xlabel("Simulation steps")
        self.ax.set_xlim(*[0, num_simulation_steps * time_interval])
        self.ax.autoscale_view(tight=True, scalex=True, scaley=True)
        self.x_ticks = np.arange(
            num_simulation_steps * time_interval, step=time_interval)

        self.lines = []  # type: List[matplotlib.lines.Line2D]
        self.num_lines = 0
        self.data_cache = np.array([])
        super(LinePlotHandler, self).__init__(plot_frequency)

    def init_lines_and_cache(self) -> None:
        """Initialise the lines on the plot and draw the y label.
        """

        self.ax.set_ylabel(self.y_label)
        for label, (linestyle, linewidth), (marker, markersize) in zip(self.line_labels,
                                                                       self.line_styles,
                                                                       self.line_markers):
            line, = self.ax.plot([], [], label=label, linestyle=linestyle, linewidth=linewidth,
                                 marker=marker, markersize=markersize)
            self.lines.append(line)

        if not self.do_testing_mode:
            plt.show()
        self.ax.legend(markerscale=2)

        self.data_cache = np.zeros((self.num_lines, self.num_simulation_steps))

    def handle(self, reporter_cache: types.ReporterCache,
               data_dict: types.DataDict, step: int) -> None:
        """Generate a live line plot, using the current state of the data dictionary, the
        reporters cache, and current time step. This method is called every time step.

        :param reporter_cache: A dictionary of items in the reporters cache.
        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        start, end = self.get_start_end(step)
        data = self.get_new_line_data(reporter_cache, data_dict, step)
        if not self.lines:
            size = data.shape[0]
            self.num_lines = size
            self.init_lines_and_cache()
            self.data_cache[:, start:end] = data
        else:
            # data_caches stores lines to speed up code
            self.data_cache[:, start:end] = data
            for j, s in enumerate(self.data_cache):
                self.lines[j].set_data(self.x_ticks[:step], s[:step])
            if not self.do_testing_mode:
                plt.pause(self.pause)
            self.ax.relim()
            self.ax.autoscale_view(tight=True, scalex=True, scaley=True)
        return

    def get_start_end(self, step) -> Tuple[int,int]:
        start = max(step - self.trigger_frequency, 0)
        end = step + 1
        assert start >= 0
        return start, end

    @property
    def line_styles(self) -> Iterable[Tuple[str, float]]:
        """Return the line styles that will automatically be used to draw the lines"""
        styles = [('-', 1) for _ in range(self.num_lines)]
        return styles

    @property
    def line_markers(self) -> List[Tuple[str, int]]:
        """Return the line markers that will automatically be used to draw the lines"""
        markers = [('', 0) for _ in range(self.num_lines)]
        return markers

    @property
    @abstractmethod
    def line_labels(self) -> Iterable[str]:
        """Return the labels that will automatically be used to label the lines"""
        pass

    @property
    @abstractmethod
    def y_label(self) -> str:
        """Return the y label that will automatically be used to label the y axis"""
        pass

    @abstractmethod
    def get_new_line_data(self, reporter_cache: types.ReporterCache, data_dict: types.DataDict,
                          step: int) -> types.Matrix:
        """Return any new line data, since the last call, in the form of a matrix
        [lines x time_steps_since_last_call]

        :param reporter_cache: A dictionary of items in the reporters cache.
        :param data_dict: The current data dictionary at this time step.
        :param step: The current step of the simulation.
        """
        pass
