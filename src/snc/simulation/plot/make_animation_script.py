import json
from functools import reduce
from tqdm import tqdm
import numpy as np

import src.snc.simulation.plot.drawing_utils as draw
import src.snc.simulation.plot.plotting_utils as plot


def stitch_data_dicts(list_of_datadicts):
    return reduce(lambda x, y: {k: x[k] + y[k] for k in x.keys()}, list_of_datadicts)


def load_json(datadict_file: str):
    """Load the datadict json file

    param: path to file
    """
    with open(datadict_file, 'r') as f:
        data_dict = json.load(f)
    return data_dict


def make_animation(name, data_dict, draw_fn, colors, time_interval, ymax=None):
    """Make an animation from a datadict and certain

    param: path to file
    """
    print('Preparing Animation')

    # specify the network to draw
    seconds = 10
    fps = 80
    progress_bar = tqdm(total=seconds * fps)
    anim = plot.plot_dual_schematic_cost_animations(data_dict, draw_fn, colors,
                                                    ymax=ymax,
                                                    do_annotations=True,
                                                    plot_cost=True,
                                                    pbar=progress_bar, width=28, height=25,
                                                    seconds=seconds, fps=fps)
    anim.save(name, fps=fps, extra_args=['-vcodec', 'libx264'])
    progress_bar.close()


if __name__ == '__main__':
    DATA_DICT1 = load_json('../demo_scripts/heuristic_datadict.json')
    DATA_DICT2 = load_json('../demo_scripts/heuristic_datadict.json')
    DATA_DICT = stitch_data_dicts([DATA_DICT1, DATA_DICT2])
    DRAW_FN = draw.draw_three_warehouses_simplified
    COLORS = {
        'buffers': [
            '#bc5d86', '#ac3267', '#501e36',
            '#91dbbe', '#70b39c', '#1c433c',
            '#70a9e2', '#7f8fd2', '#8f74b8'
        ],
        'suppliers': ['#cc87a5', '#befcda', '#6ec1ea'],
        'neutral': '#000000'
    }
    TIME_INTERVAL = 1
    make_animation('tri_test.mp4', [DATA_DICT], DRAW_FN, COLORS, TIME_INTERVAL, ymax=-np.inf)
