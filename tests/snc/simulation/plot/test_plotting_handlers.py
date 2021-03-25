import numpy as np
import snc.simulation.plot.plotting_handlers as ph


def test_cumulative_cost_plotter_non_discounted_large_plot_frequency():
    plotter = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                       plot_frequency=10, discounted=False)

    cost = [np.array([1]), np.array([10]), np.array([20])]
    data_dict = {'cost': cost}
    reporter_cache = {'discount_factor': [np.array([[0.5]])]}
    step = 2

    for s in range(step):
        plotter.handle(data_dict=data_dict, reporter_cache=reporter_cache, step=s)

    cumul = plotter.get_new_line_data(data_dict=data_dict, reporter_cache=reporter_cache, step=step)
    assert np.all(cumul == np.array([1, 11, 31]))


def test_cumulative_cost_plotter_discounted_large_plot_frequency():
    plotter = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                       plot_frequency=10, discounted=True)

    cost = [np.array([1]), np.array([10]), np.array([20])]
    data_dict = {'cost': cost}
    reporter_cache = {'discount_factor': [np.array([[0.5]])]}
    step = 2

    for s in range(step):
        plotter.handle(data_dict=data_dict, reporter_cache=reporter_cache, step=s)

    cumul = plotter.get_new_line_data(data_dict=data_dict, reporter_cache=reporter_cache, step=step)
    assert np.all(cumul == np.array([1, 6, 11]))


def test_cumulative_cost_plotter_non_discounted_small_plot_frequency():
    plotter = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                       plot_frequency=2, discounted=False)

    cost = [np.array([1]), np.array([10]), np.array([20])]
    data_dict = {'cost': cost}
    reporter_cache = {'discount_factor': [np.array([[0.5]])]}
    step = 2

    for s in range(step):
        plotter.handle(data_dict=data_dict, reporter_cache=reporter_cache, step=s)

    cumul = plotter.get_new_line_data(data_dict=data_dict, reporter_cache=reporter_cache, step=step)
    assert np.all(cumul == np.array([1, 11, 31]))


def test_cumulative_cost_plotter_discounted_small_plot_frequency():
    plotter = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                       plot_frequency=2, discounted=True)

    cost = [np.array([1]), np.array([10]), np.array([20])]
    data_dict = {'cost': cost}
    reporter_cache = {'discount_factor': [np.array([[0.5]])]}
    step = 2

    for s in range(step):
        plotter.handle(data_dict=data_dict, reporter_cache=reporter_cache, step=s)

    cumul = plotter.get_new_line_data(data_dict=data_dict, reporter_cache=reporter_cache, step=step)
    assert np.all(cumul == np.array([1, 6, 11]))


def test_cumulative_cost_plotter_label():
    plotter1 = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                        plot_frequency=2, discounted=True)
    assert plotter1.y_label == 'Cumulative Disc Cost'
    assert [*plotter1.line_labels] == ['Total Cost']

    plotter2 = ph.CumulativeCostPlotter(num_simulation_steps=10, time_interval=1.0,
                                        plot_frequency=2, discounted=False)
    assert plotter2.y_label == 'Cumulative Cost'
    assert [*plotter2.line_labels] == ['Total Cost']
