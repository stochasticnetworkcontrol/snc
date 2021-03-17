from copy import deepcopy
import cvxpy as cvx
import json
import numpy as np
import re
from typing import Any, Dict


class NumpyEncoder(json.JSONEncoder):
    """
    A JSON encoder that can handle numpy objects and convert to json.
    """
    # Example taken from python.org docs
    def default(self, o): # pylint: disable=E0202
        if hasattr(o, 'to_serializable'):
            return o.to_serializable()
        elif isinstance(o, set):
            return list(o)
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.random.RandomState):
            return str(o)
        return json.JSONEncoder.default(self, o)


def format_json_with_np(string: str) -> str:
    """
    JSON dumps all items on new lines. This regex makes a string of indented json put innermost
    arrays on same line.

    :param string: a string of json.
    :return: s: string of indented json.
    """
    s = re.sub(r'([-0-9],)\n\s*(?=[-0-9])', r'\1 ', string)
    s = re.sub(r'\[\n\s*([-0-9])', r'[\1', s)
    s = re.sub(r'([-0-9])\n\s*\]', r'\1]', s)
    return s


def clean_to_serializable(self: Any) -> Dict:
    d = deepcopy(self.__dict__)
    return iterate_dict(d)


def is_not_valid_json_value(value) -> bool:
    invalid_instances = (
        cvx.Parameter,
        cvx.Problem,
        cvx.Variable,
        cvx.constraints.Inequality,
        cvx.constraints.Equality,
        type
    )
    return isinstance(value, invalid_instances) or callable(value)


def iterate_dict(o):
    d = deepcopy(o)
    for key, value in o.items():
        if is_not_valid_json_value(value):
            d.pop(key)
        elif isinstance(value, dict):
            d[key] = iterate_dict(value)
        elif isinstance(value, list):
            d[key] = iterate_list(value)
    return d


def iterate_list(o):
    clr = []
    for i, value in enumerate(o):
        if is_not_valid_json_value(value):
            clr.append(i)
        elif isinstance(value, list):
            o[i] = iterate_list(value)
        elif isinstance(value, dict):
            o[i] = iterate_dict(value)
    clr.sort(reverse=True)
    for i in clr:
        o.pop(i)
    return o
