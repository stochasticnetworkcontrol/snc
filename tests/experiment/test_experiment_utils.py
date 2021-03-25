import argparse

from src.experiment import experiment_utils


def test_param_string_creation():
    """Tests that arguments are formatted correctly in the string used for logging."""
    # This example is based on arguments to command_builder.py
    params = argparse.Namespace(
        docker_image='snc/test',
        entry_point='/code/snc/experiment/rl/rl_experiment_script.py',
        config='example_list_config.json'
    )

    param_string = experiment_utils.get_args_string(params)

    expected_param_string = \
        '      config: example_list_config.json\n'\
        'docker_image: snc/test\n'\
        ' entry_point: /code/snc/experiment/rl/rl_experiment_script.py\n'

    assert param_string == expected_param_string


def test_flatten_dict_simple():
    nested_dict = {
        'a': 1,
        'b': {
            'c': ['d', 'e', 'f'],
            'g': 3
        },
        'h': (2, 1)
    }
    flat_dict = experiment_utils.flatten_dict(nested_dict, separator=' ')
    assert flat_dict == {
        'a': 1,
        'b c': ['d', 'e', 'f'],
        'b g': 3,
        'h': (2, 1)
    }


def test_flatten_dict_complex():
    nested_dict = {
        'a': 1,
        'b': {
            'c': {'d': 1, 'e': 2, 'f': {'g': 3}},
            'h': 4
        },
        'i': (3, 2, 1)
    }
    flat_dict = experiment_utils.flatten_dict(nested_dict, separator=' ')
    assert flat_dict == {
        'a': 1,
        'b c d': 1,
        'b c e': 2,
        'b c f g': 3,
        'b h': 4,
        'i': (3, 2, 1)
    }


def test_unflatten_dict_simple():
    flat_dict = {
        'a': 1,
        'b c': ['d', 'e', 'f'],
        'b g': 3,
        'h': (2, 1)
    }
    nested_dict = experiment_utils.unflatten_dict(flat_dict, separator=' ')
    assert nested_dict == {
        'a': 1,
        'b': {
            'c': ['d', 'e', 'f'],
            'g': 3
        },
        'h': (2, 1)
    }


def test_unflatten_dict_complex():
    flat_dict = {
        'a': 1,
        'b c d': 1,
        'b c e': 2,
        'b c f g': 3,
        'b h': 4,
        'i': (3, 2, 1)
    }
    nested_dict = experiment_utils.unflatten_dict(flat_dict, separator=' ')
    assert nested_dict == {
        'a': 1,
        'b': {
            'c': {'d': 1, 'e': 2, 'f': {'g': 3}},
            'h': 4
        },
        'i': (3, 2, 1)
    }

