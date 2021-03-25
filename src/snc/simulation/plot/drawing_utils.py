from typing import Optional, Dict, List

from matplotlib import axes
from matplotlib import text as mtext
from matplotlib import patches as mpatches

from src import snc as types


################
# DRAW UNITS
################

def draw_buffer(ax: axes.Axes, coord: types.Coord2D, num_items: int,
                max_capacity: int, color: str, do_annotations: bool, do_reverse: bool = False,
                is_extra_high: Optional[bool] = False) -> List[mpatches.Patch]:
    """Draw a horizontal buffer diagram on axes, at a certain coordinate,
    with items added from the left ('top' is on the left.)

    :param ax: matplotlib axes on which to draw
    :param coord: coordinate at which to place buffer (bottom left of it)
    :param num_items: number of items in the buffer (state)
    :param max_capacity: max capacity of the buffer
    :param color: color of the buffer
    :param do_annotations: whether to annotate the buffer with the current state
    :param do_reverse: if true, draw the buffer in reverse (items added from right)
    :param is_extra_high: if true, draw a buffer that is twice the height of the default option
    """

    b_width = 3
    if is_extra_high:
        b_height = 2
    else:
        b_height = 1
    p1 = mpatches.Rectangle(coord, b_width, b_height, color=color)
    fill_fraction = num_items / max_capacity
    f_width = (1 - fill_fraction) * b_width
    ax.add_patch(p1)

    if do_reverse:
        white_coord = (coord[0] + b_width - f_width, coord[1])
        p2 = mpatches.Rectangle(white_coord, f_width, b_height, color='w')
    else:
        p2 = mpatches.Rectangle(coord, f_width, b_height, color='w')
    ax.add_patch(p2)

    p3 = mpatches.Rectangle(coord, b_width, b_height, fill=False, color='k')
    ax.add_patch(p3)

    if do_annotations:
        msg = '{:.0f}'.format(num_items)
        text_coord = [coord[0] + .2, coord[1] + b_height + 0.2]
        ax.annotate(msg, coord, text_coord, size=14)

    return [p1, p2, p3]


def draw_station(ax: axes.Axes, coord: types.Coord2D, buffers: int, color: str,
                 active_b: Optional[int] = None) -> List[mpatches.Patch]:
    """Draw a vertical station which holds a resource.

    :param ax: matplotlib axes on which to draw
    :param coord: coordinate at which to place station (bottom left of it)
    :param buffers: number of buffers attached to the station
    :param color: color of the station
    :param active_b: which buffer/action is active (where the resource should be)
    """
    s_width = 1
    s_height = buffers + (buffers - 1) * 2

    margin_height = 0.5

    rec_coord = [coord[0], coord[1] - margin_height]
    p1 = mpatches.Rectangle(rec_coord, s_width, s_height +
                            2 * margin_height, fill=False, color='k')
    ax.add_patch(p1)

    if active_b is None:
        centre = [coord[0] + s_width / 2 + 0.02, coord[1] + s_height / 2]
    else:
        b = active_b - 1
        centre = [coord[0] + s_width / 2 + 0.02, coord[1] + 0.5 + 3 * b]

    p2 = mpatches.Circle(centre, s_width * .42, fill=False, color='k')
    ax.add_patch(p2)
    p3 = mpatches.Circle(centre, s_width * .4, color=color, alpha=0.4)
    ax.add_patch(p3)
    return [p1, p2, p3]


def draw_demand(ax: axes.Axes, coord: types.Coord2D, color: str,
                active: Optional[bool] = None) -> List[mpatches.Patch]:
    '''Draw a demand/arrivals source

    :param ax: matplotlib axes on which to draw
    :param coord: coordinate at which to place demand source (bottom left of it)
    :param color: color of the demand source
    :param active: whether demand is active (are items arriving)
    '''
    d_width = 1
    d_height = 1
    alpha = 0.3 if not active else .8

    centre = [coord[0] + d_width / 2 + 0.02, coord[1] + d_height / 2]

    p1 = mpatches.RegularPolygon(
        centre, 7, radius=0.5, color=color, alpha=alpha)
    ax.add_patch(p1)
    p2 = mpatches.RegularPolygon(
        centre, 7, radius=0.52, fill=False, color='k', )
    ax.add_patch(p2)
    return [p1, p2]


def draw_straight_arrows(ax: axes.Axes, from_coord: types.Coord2D, to_coord: types.Coord2D,
                         color: str, active: bool, effect: int,
                         do_annotations: bool, no_head: bool = False) -> List[mpatches.Patch]:
    """Draw a straight arrow between two points, indicating an activity.

    :param ax: matplotlib axes on which to draw
    :param from_coord: point at the tail of the arrow (bottom left of it)
    :param to_coord: point at the head of the arrow (bottom left of it)
    :param color: color of the arrow when active
    :param active: whether arrow is active (are items arriving)
    :param effect: what value/no of items are being processed along the arrow
    :param do_annotations: whether to annotate the arrow with the no of items
        being processd
    :param no_head: whether do suppress a head to arrow
    """
    alpha = 0.3 if not active else 1
    color = 'k' if not active else color
    if no_head:
        a_style = '-'
        do_annotations = False
    else:
        a_style = mpatches.ArrowStyle("-|>", head_length=7, head_width=4)
    mid = 0.5
    mid_from_coord = [from_coord[0], from_coord[1] + mid]
    mid_to_coord = [to_coord[0], to_coord[1] + mid]
    p1 = mpatches.FancyArrowPatch(
        mid_from_coord, mid_to_coord, arrowstyle=a_style, alpha=alpha, color=color)
    ax.add_patch(p1)

    if do_annotations and active:
        msg = '{:.0f}'.format(abs(effect))
        text_coord = [(from_coord[0] + to_coord[0]) * .5 - 0.6, to_coord[1] + 1.]
        ax.annotate(msg, from_coord, text_coord, size=14, color=color)

    return [p1]


def draw_reset(ax: axes.Axes, width: float, height: float) -> None:
    """
    Fully reset a plot by deleting all annotations and patches, and drawing a
    blank background

    :param ax: matplotlib axes on which to draw
    :param width: the width of the plot
    :param height: the height of the plot
    """
    for child in ax.get_children():
        if isinstance(child, mtext.Annotation):
            child.remove()
    for p in reversed(ax.patches):
        p.remove()

    p1 = mpatches.Rectangle((0, 0), width, height, fill=True, color='w')
    ax.add_patch(p1)


def annotate_cost_and_time(ax: axes.Axes, time_step: Optional[int] = None,
                           cumul_cost: Optional[float] = None) -> None:
    """
    Annotate the cost and time step at the bottom left of the drawing plot

    :param ax: matplotlib axes on which to draw
    :param time_step: current time step
    :param cumul_cost: the cumulative cost at the current time step
    """
    msg = ''
    if time_step is not None:
        msg += '\ntime: {}'.format(time_step)
    if cumul_cost is not None:
        msg += '\ndiscounted total cost: {:.0f}'.format(cumul_cost)
    ax.annotate(msg, (0, -1), color='k', size=14)


#################
# DRAW COMPOSITE
#################

def draw_reentrant_line(ax: axes.Axes, state: types.StateSpace, action,
                        demand: types.StateSpace, effects: types.StateSpace, max_capacity: int,
                        colors: Dict, time_step: Optional[int] = None,
                        cumul_cost: Optional[float] = None, do_annotations: bool = True) -> None:
    """A drawn example of the simple reentrant line (standard example) at a single
    time step.

    :param ax: matplotlib axes on which to draw
    :param state: a vector with the current state (X_t)
    :param action: a vector with the current actions (U_t)
    :param demand: a vector with the changes due to demand/arrivals (A_t+1)
    :param effects: a vector with the changes due to processing (B_t+1 * U_t)
    :param max_capacity: the max capacity of the buffers
    :param colors: a dictionary of colors - {'buffers':['', ...], 'demand': '', 'neutral': ''}
    :param time_step: the current time step
    :param cumul_cost: the current cumulative cost
    :param do_annotations: whether to put annotations of numbers for current states,
        demand, and arrivals
    """
    assert len(colors['buffers']) == len(action) == len(state)
    demand_color = colors['demand']
    neutral_color = colors['neutral']

    draw_buffer(ax, (5, 5), state[0], max_capacity,
                colors['buffers'][0], do_annotations)
    draw_buffer(ax, (13, 5), state[1], max_capacity,
                colors['buffers'][1], do_annotations)
    draw_buffer(ax, (9, 2), state[2], max_capacity,
                colors['buffers'][2], do_annotations, do_reverse=True)

    if action[2] and not action[0]:
        s1_action = 1
        s1_col = colors['buffers'][2]
    elif action[0] and not action[2]:
        s1_action = 2
        s1_col = colors['buffers'][0]
    else:
        s1_action = None
        s1_col = neutral_color
    draw_station(ax, (8, 2), 2, s1_col, s1_action)

    if action[1]:
        s2_col = colors['buffers'][1]
        s2_action = 1
    else:
        s2_col = neutral_color
        s2_action = None
    draw_station(ax, (16, 5), 1, s2_col, s2_action)

    draw_demand(ax, (2, 5), demand_color, active=demand[0])
    draw_straight_arrows(ax, (3, 5), (5, 5), color=demand_color, active=demand[0],
                         effect=demand[0], do_annotations=do_annotations)

    draw_straight_arrows(ax, (8, 5), (13, 5), color=colors['buffers'][0], active=action[0],
                         effect=effects[0], do_annotations=do_annotations)
    draw_straight_arrows(ax, (16, 5), (19, 5), color=colors['buffers'][1], active=action[1],
                         effect=effects[1], do_annotations=do_annotations, no_head=True)
    draw_straight_arrows(ax, (18.9, 5.1), (18.9, 1.9), color=colors['buffers'][1],
                         active=action[1], effect=effects[1], do_annotations=do_annotations,
                         no_head=True)
    draw_straight_arrows(ax, (19, 2), (12, 2), color=colors['buffers'][1], active=action[1],
                         effect=effects[1], do_annotations=do_annotations)
    draw_straight_arrows(ax, (9, 2), (2, 2), color=colors['buffers'][2], active=action[2],
                         effect=effects[2], do_annotations=do_annotations)

    annotate_cost_and_time(ax, time_step, cumul_cost)


def draw_double_reentrant_line_only_shared_resources(ax: axes.Axes, state: types.StateSpace, action,
                                                     demand: types.StateSpace,
                                                     effects: types.StateSpace, max_capacity: int,
                                                     colors: Dict, time_step: Optional[int] = None,
                                                     cumul_cost: Optional[float] = None,
                                                     do_annotations: bool = True) -> None:
    """A drawn example of the simple reentrant line (standard example) at a single
    time step.
    :param ax: matplotlib axes on which to draw
    :param state: a vector with the current state (X_t)
    :param action: a vector with the current actions (U_t)
    :param demand: a vector with the changes due to demand/arrivals (A_t+1)
    :param effects: a vector with the changes due to processing (B_t+1 * U_t)
    :param max_capacity: the max capacity of the buffers
    :param colors: a dictionary of colors - {'buffers':['', ...], 'demand': '', 'neutral': ''}
    :param time_step: the current time step
    :param cumul_cost: the current cumulative cost
    :param do_annotations: whether to put annotations of numbers for current states,
        demand, and arrivals
    """
    assert len(colors['buffers']) == len(state)
    demand_color = colors['demand']
    neutral_color = colors['neutral']
    draw_buffer(ax, (4, 4), state[0], max_capacity,
                colors['buffers'][0], do_annotations)
    draw_buffer(ax, (12, 4), state[1], max_capacity,
                colors['buffers'][1], do_annotations)
    draw_buffer(ax, (16, 1), state[2], max_capacity,
                colors['buffers'][2], do_annotations, do_reverse=True)
    draw_buffer(ax, (8, 1), state[3], max_capacity,
                colors['buffers'][3], do_annotations, do_reverse=True)
    if action[3] and not action[0]:
        s1_action = 1
        s1_col = colors['buffers'][3]
    elif action[0] and not action[3]:
        s1_action = 2
        s1_col = colors['buffers'][0]
    else:
        s1_action = None
        s1_col = neutral_color
    draw_station(ax, (7, 1), 2, s1_col, s1_action)
    if action[2] and not action[1]:
        s2_action = 1
        s2_col = colors['buffers'][2]
    elif action[1] and not action[2]:
        s2_action = 2
        s2_col = colors['buffers'][1]
    else:
        s2_action = None
        s2_col = neutral_color
    draw_station(ax, (15, 1), 2, s2_col, s2_action)
    draw_demand(ax, (1, 4), demand_color, active=demand[0])
    draw_straight_arrows(ax, (2, 4), (4, 4), color=demand_color, active=demand[0],
                         effect=demand[0], do_annotations=do_annotations)
    draw_straight_arrows(ax, (8, 4), (12, 4), color=colors['buffers'][0], active=action[0],
                         effect=effects[0], do_annotations=do_annotations)
    draw_straight_arrows(ax, (16, 4), (21, 4), color=colors['buffers'][1],
                         active=action[1], effect=effects[1], do_annotations=do_annotations,
                         no_head=True)
    draw_straight_arrows(ax, (20.9, 4.1), (20.9, 0.9), color=colors['buffers'][1],
                         active=action[1], effect=effects[1], do_annotations=do_annotations,
                         no_head=True)
    draw_straight_arrows(ax, (21, 1), (19, 1), color=colors['buffers'][1], active=action[1],
                         effect=effects[1], do_annotations=do_annotations)
    draw_straight_arrows(ax, (15, 1), (11, 1), color=colors['buffers'][2], active=action[2],
                         effect=effects[2], do_annotations=do_annotations)
    draw_straight_arrows(ax, (7, 1), (1, 1), color=colors['buffers'][3], active=action[3],
                         effect=effects[3], do_annotations=do_annotations)
    annotate_cost_and_time(ax, time_step, cumul_cost)


def draw_three_warehouses_simplified(ax: axes.Axes, state: types.StateSpace, action,
                                     demand: types.StateSpace,
                                     effects: types.StateSpace, max_capacity: int,
                                     colors: Dict, time_step: Optional[int] = None,
                                     cumul_cost: Optional[float] = None,
                                     do_annotations: bool = True) -> None:
    """A drawn the distribution with rebalancing simplified example with 3 warehouses.
    :param ax: matplotlib axes on which to draw
    :param state: a vector with the current state (X_t)
    :param action: a vector with the current actions (U_t)
    :param demand: a vector with the changes due to demand/arrivals (A_t+1)
    :param effects: a vector with the changes due to processing (B_t+1 * U_t)
    :param max_capacity: the max capacity of the buffers
    :param colors: a dictionary of colors - {'buffers':['', ...], 'demand': '', 'neutral': ''}
    :param time_step: the current time step
    :param cumul_cost: the current cumulative cost
    :param do_annotations: whether to put annotations of numbers for current states,
        demand, and arrivals
    """
    assert len(colors['buffers']) == len(state)
    suppliers_color = colors['suppliers']

    draw_buffer(ax, (8, 17), state[0], max_capacity,
                colors['buffers'][0], do_annotations, is_extra_high=True)
    draw_buffer(ax, (17, 17.5), state[1], max_capacity,
                colors['buffers'][1], do_annotations)
    draw_buffer(ax, (21, 17.5), state[2], max_capacity,
                colors['buffers'][2], do_annotations, do_reverse=True)

    draw_buffer(ax, (8, 10), state[3], max_capacity,
                colors['buffers'][3], do_annotations, is_extra_high=True)
    draw_buffer(ax, (17, 10.5), state[4], max_capacity,
                colors['buffers'][4], do_annotations)
    draw_buffer(ax, (21, 10.5), state[5], max_capacity,
                colors['buffers'][5], do_annotations, do_reverse=True)

    draw_buffer(ax, (8, 3), state[6], max_capacity,
                colors['buffers'][6], do_annotations, is_extra_high=True)
    draw_buffer(ax, (17, 3.5), state[7], max_capacity,
                colors['buffers'][7], do_annotations)
    draw_buffer(ax, (21, 3.5), state[8], max_capacity,
                colors['buffers'][8], do_annotations, do_reverse=True)

    if action[0] or action[3] or action[12]:
        s_col = colors['buffers'][0]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (11, 17.5), 1, s_col, s_action)

    if action[4] or action[7] or action[13]:
        s_col = colors['buffers'][3]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (11, 10.5), 1, s_col, s_action)

    if action[8] or action[11] or action[14]:
        s_col = colors['buffers'][6]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (11, 3.5), 1, s_col, s_action)

    if action[1]:
        s_col = colors['buffers'][1]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (20, 17.5), 1, s_col, s_action)

    if action[5]:
        s_col = colors['buffers'][4]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (20, 10.5), 1, s_col, s_action)

    if action[9]:
        s_col = colors['buffers'][7]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (20, 3.5), 1, s_col, s_action)

    if action[2]:
        s_col = suppliers_color[0]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (2, 17.5), 1, s_col, s_action)
    if action[6]:
        s_col = suppliers_color[1]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (2, 10.5), 1, s_col, s_action)
    if action[10]:
        s_col = suppliers_color[2]
        s_action = 1
    else:
        s_col = 'k'
        s_action = None
    draw_station(ax, (2, 3.5), 1, s_col, s_action)

    # arrows of the top warehouse lane
    draw_straight_arrows(ax, (12, 17.5), (17, 17.5), color=colors['buffers'][0], active=action[0],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (24, 17.5), (26, 17.5), color=colors['buffers'][2], active=action[1],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (3, 17.50), (8, 17.50), color=suppliers_color[0], active=action[2],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (12, 17), (13.1, 17), color=colors['buffers'][0],
                         active=action[3], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13, 17.1), (13, 15.4), color=colors['buffers'][0],
                         active=action[3], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13.1, 15.5), (5.9, 15.5), color=colors['buffers'][0],
                         active=action[3], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6, 15.6), (6, 11.4), color=colors['buffers'][0],
                         active=action[3], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (5.9, 11.5), (8, 11.5), color=colors['buffers'][0], active=action[3],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (12, 18), (15.1, 18), color=colors['buffers'][0],
                         active=action[12], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (15, 18.1), (15, 1.4), color=colors['buffers'][0],
                         active=action[12], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (15.1, 1.5), (6.9, 1.5), color=colors['buffers'][0],
                         active=action[12], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (7, 1.4), (7, 2.6), color=colors['buffers'][0],
                         active=action[12], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6.9, 2.5), (8, 2.5), color=colors['buffers'][0], active=action[12],
                         effect=0, do_annotations=False)

    # arrows of middle warehouse lane
    draw_straight_arrows(ax, (12, 10.5), (17, 10.5), color=colors['buffers'][3], active=action[4],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (24, 10.5), (26, 10.5), color=colors['buffers'][5], active=action[5],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (3, 10.50), (8, 10.50), color=suppliers_color[1], active=action[6],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (12, 10), (13.1, 10), color=colors['buffers'][3],
                         active=action[7], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13, 10.1), (13, 8.4), color=colors['buffers'][3],
                         active=action[7], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13.1, 8.5), (5.9, 8.5), color=colors['buffers'][3],
                         active=action[7], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6, 8.6), (6, 4.4), color=colors['buffers'][3],
                         active=action[7], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (5.9, 4.5), (8, 4.5), color=colors['buffers'][3], active=action[7],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (12, 11), (13.1, 11), color=colors['buffers'][3],
                         active=action[13], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13, 10.9), (13, 12.6), color=colors['buffers'][3],
                         active=action[13], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13.1, 12.5), (6.9, 12.5), color=colors['buffers'][3],
                         active=action[13], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (7, 12.4), (7, 16.6), color=colors['buffers'][3],
                         active=action[13], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6.9, 16.5), (8, 16.5), color=colors['buffers'][3], active=action[13],
                         effect=0, do_annotations=False)

    # arrows of the bottom warehouse lane
    draw_straight_arrows(ax, (12, 3.5), (17, 3.5), color=colors['buffers'][6], active=action[8],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (24, 3.5), (26, 3.5), color=colors['buffers'][8], active=action[9],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (3, 3.50), (8, 3.50), color=suppliers_color[2], active=action[10],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (11.9, 3), (14.1, 3), color=colors['buffers'][6],
                         active=action[11], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (14, 2.9), (14, 19.6), color=colors['buffers'][6],
                         active=action[11], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (14.1, 19.5), (5.9, 19.5),color=colors['buffers'][6],
                         active=action[11], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6, 19.6), (6, 18.4), color=colors['buffers'][6],
                         active=action[11], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (5.9, 18.5), (8, 18.5), color=colors['buffers'][6], active=action[11],
                         effect=0, do_annotations=False)

    draw_straight_arrows(ax, (12, 4), (13.1, 4), color=colors['buffers'][6],
                         active=action[14], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13, 3.9), (13, 5.6), color=colors['buffers'][6],
                         active=action[14], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (13.1, 5.5), (6.9, 5.5), color=colors['buffers'][6],
                         active=action[14], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (7, 5.4), (7, 9.6), color=colors['buffers'][6],
                         active=action[14], effect=0, do_annotations=False,
                         no_head=True)
    draw_straight_arrows(ax, (6.9, 9.5), (8, 9.5), color=colors['buffers'][6], active=action[14],
                         effect=0, do_annotations=False)

    annotate_cost_and_time(ax, time_step, cumul_cost)
