from copy import deepcopy

import cvxpy as cvx
import json
import numpy as np

import snc.simulation.store_data.numpy_encoder as numpy_encoder


class MySerializableClass:
    def to_serializable(self):
        return {"hello": "world"}


def test_json_encoder_with_numpy():
    x = [10, 20, 30, 40, 50]
    a = np.array(x)

    s = json.dumps(a, cls=numpy_encoder.NumpyEncoder)
    assert s == str(x)


def test_json_encoder_with_custom_class():
    obj = MySerializableClass()

    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder)
    assert '{"hello": "world"}' == s


def test_json_encoder_with_deep_custom_class():
    obj = MySerializableClass()
    x = [10, 20, 30, 40, 50]
    a = np.array(x)

    obj = {'my_class': obj, 'my_array': a}
    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder, sort_keys=True)
    assert '{"my_array": [10, 20, 30, 40, 50], "my_class": {"hello": "world"}}' == s


def test_json_formatter_1():
    raw = '''
{
    "asymptotic_covariance_params": [
        5,
        100,
        1000
    ],
    "asymptotic_workload_cov": [
        [
            24.20047888797887,
            20.736179183444843
        ],
        [
            20.736179183444843,
            27.975200481243053
        ]
    ],
    "beta_star_cone_list": [
        [
            8.053423193500828
        ]
    ]
}'''
    formatted = '''
{
    "asymptotic_covariance_params": [5, 100, 1000],
    "asymptotic_workload_cov": [
        [24.20047888797887, 20.736179183444843],
        [20.736179183444843, 27.975200481243053]
    ],
    "beta_star_cone_list": [
        [8.053423193500828]
    ]
}'''

    assert formatted == numpy_encoder.format_json_with_np(raw)
    pass


def test_json_formatter_2():
    raw = '''
{
    "asymptotic_workload_cov": [
        [
            -21.20047888797887,
            23.20047888797887,
            24.20047888797887,
            25.20047888797887,
            26.20047888797887,
            27.20047888797887
        ],
        [
            31.20047888797887,
            33.20047888797887,
            34.20047888797887,
            35.20047888797887,
            36.20047888797887,
            37.20047888797887
        ],
        [
            -41.20047888797887,
            43.20047888797887,
            44.20047888797887,
            45.20047888797887,
            46.20047888797887,
            47.20047888797887
        ],
        [
            51.20047888797887,
            53.20047888797887,
            54.20047888797887,
            55.20047888797887,
            56.20047888797887,
            57.20047888797887
        ],
        [
            61.20047888797887,
            63.20047888797887,
            64.20047888797887,
            65.20047888797887,
            66.20047888797887,
            67.20047888797887
        ]
    ]
}'''

    formatted = '''
{
    "asymptotic_workload_cov": [
        [-21.20047888797887, 23.20047888797887, 24.20047888797887, 25.20047888797887, 26.20047888797887, 27.20047888797887],
        [31.20047888797887, 33.20047888797887, 34.20047888797887, 35.20047888797887, 36.20047888797887, 37.20047888797887],
        [-41.20047888797887, 43.20047888797887, 44.20047888797887, 45.20047888797887, 46.20047888797887, 47.20047888797887],
        [51.20047888797887, 53.20047888797887, 54.20047888797887, 55.20047888797887, 56.20047888797887, 57.20047888797887],
        [61.20047888797887, 63.20047888797887, 64.20047888797887, 65.20047888797887, 66.20047888797887, 67.20047888797887]
    ]
}'''
    assert formatted == numpy_encoder.format_json_with_np(raw)


def test_json_formatter_3():
    raw = '''
{
    "map_action_to_index": {
        "act1": 0,
        "act11": 7,
        "act13": 8,
        "act14": 9,
        "act15": 10,
        "act17": 11,
    "num_actions": 15,
    "safety_stocks": [
        10,
        5,
        0,
        10,
        5,
        0,
        10,
        5,
        0
    ]
}'''
    formatted = '''
{
    "map_action_to_index": {
        "act1": 0,
        "act11": 7,
        "act13": 8,
        "act14": 9,
        "act15": 10,
        "act17": 11,
    "num_actions": 15,
    "safety_stocks": [10, 5, 0, 10, 5, 0, 10, 5, 0]
}'''
    assert formatted == numpy_encoder.format_json_with_np(raw)


def test_json_formatter_no_op():
    raw = '''
{
    "capacity": [
        [
            Infinity
        ],
        [
            Infinity
        ]
    ]
}'''
    assert raw == numpy_encoder.format_json_with_np(raw)


def test_json_formatter_integer():
    raw = '''
{
    "asymptotic_covariance_params": [
        5,
        100,
        1000
    ]
}'''
    formatted = '''
{
    "asymptotic_covariance_params": [5, 100, 1000]
}'''
    assert formatted == numpy_encoder.format_json_with_np(raw)


def test_json_formatter_scientific():
    raw = '''
{
    "asymptotic_covariance_params": [
        1e5,
        2e100,
        1.3e1000
    ]
}'''
    formatted = '''
{
    "asymptotic_covariance_params": [1e5, 2e100, 1.3e1000]
}'''
    assert formatted == numpy_encoder.format_json_with_np(raw)


class MyDeepSerializableClass:
    def __init__(self, properties):
        self.properties = properties

    def to_serializable(self):
        return numpy_encoder.clean_to_serializable(self)


def test_clean_to_serializable_shallow_dict():
    param = cvx.Parameter()
    var = cvx.Variable()
    prob = cvx.Problem(objective=cvx.Minimize(2*var))
    properties = {
        'a': 1,
        'cvx_param': param,
        'cvx_prob': prob,
        'cvx_var': var,
        'type': MyDeepSerializableClass,
        'func': numpy_encoder.clean_to_serializable,
        'b': 2
    }
    obj = MyDeepSerializableClass(properties)
    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder)
    assert '{"properties": {"a": 1, "b": 2}}' == s


def test_clean_to_serializable_shallow_list():
    param = cvx.Parameter()
    var = cvx.Variable()
    prob = cvx.Problem(objective=cvx.Minimize(2*var))
    properties = [
        1,
        param,
        prob,
        2,
        var,
        MyDeepSerializableClass,
        numpy_encoder.clean_to_serializable,
        3
    ]
    obj = MyDeepSerializableClass(properties)
    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder)
    assert '{"properties": [1, 2, 3]}' == s


def test_clean_to_serializable_deep_dict():
    param = cvx.Parameter()
    var = cvx.Variable()
    prob = cvx.Problem(objective=cvx.Minimize(2*var))
    properties = {
        'a': 1,
        'cvx_param': param,
        'cvx_prob': prob,
        'cvx_var': var,
        'type': MyDeepSerializableClass,
        'func': numpy_encoder.clean_to_serializable,
        'b': 2,
    }
    deep_prop = deepcopy(properties)
    deep_prop['b'] = deepcopy(properties)
    obj = MyDeepSerializableClass(deep_prop)
    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder)
    assert '{"properties": {"a": 1, "b": {"a": 1, "b": 2}}}' == s


def test_clean_to_serializable_deep_list():
    param = cvx.Parameter()
    var = cvx.Variable()
    prob = cvx.Problem(objective=cvx.Minimize(2*var))
    properties = [
        1,
        param,
        prob,
        2,
        var,
        MyDeepSerializableClass,
        numpy_encoder.clean_to_serializable,
        3
    ]
    deep_prop = deepcopy(properties)
    deep_prop.append(properties)
    obj = MyDeepSerializableClass(deep_prop)
    s = json.dumps(obj, cls=numpy_encoder.NumpyEncoder)
    assert '{"properties": [1, 2, 3, [1, 2, 3]]}' == s
