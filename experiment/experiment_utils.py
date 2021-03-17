import argparse
from datetime import datetime
from collections import OrderedDict
from typing import Any, Dict

from warnings import warn

# TODO: write test
# type info
# installation setup


def get_args_string(args: argparse.Namespace) -> str:
    """
    Creates a string summarising the argparse arguments.
    :param args: parser.parse_args()

    :return: String of the arguments of the argparse namespace.
    """
    string = ''
    if hasattr(args, 'experiment_name'):
        string += f'{args.experiment_name} ({datetime.now()})\n'
    max_length = max([len(k) for k, _ in vars(args).items()])
    new_dict = OrderedDict((k, v) for k, v in sorted(
        vars(args).items(), key=lambda x: x[0]
    ))
    for key, value in new_dict.items():
        string += ' ' * (max_length - len(key)) + key + ': ' + str(value) + '\n'
    return string


def flatten_dict(
        nested_dict: Dict[str, Any],
        separator: str = ' ',
        prefix: str = ''
    ) -> Dict[str, Any]:
    """
    Takes a (possibly) nested dictionary and flattens it so that there is only one layer of keys.
    The keys of the flattened dictionary will be of the form 'parent_key + separator + child_key',
    with possibly multiple layers of nesting.

    :param nested_dict: The nested dictionary to flatten.
    :param separator: The separator to use in flattening the dictionary, be careful not to use
        something which may already be in a key such as an _.
    :param prefix: Used in the recursion only.
    :return: A flattened dictionary whose keys reflect the structure of the dictionary originally
        passed in.
    """
    # Use a dictionary comprehension with recursion to attain the flattened dictionary desired.
    # Code adapted from the link below.
    # https://www.geeksforgeeks.org/python-convert-nested-dictionary-into-flattened-dictionary
    return {prefix + separator + k if prefix else k: v
            for kk, vv in nested_dict.items()
            for k, v in flatten_dict(vv, separator, kk).items()
            } if isinstance(nested_dict, dict) else {prefix: nested_dict}


def unflatten_dict(
        flat_dict: Dict[str, Any],
        separator: str = ' '
    ) -> Dict[str, Any]:
    """
    The inverse operator to flatten_dict.
    Takes a flattened dictionary and returns a nested one.

    :param flat_dict: The dictionary to be nested.
    :param separator: The separator used when flattening the dictionary to separate keys of
        different levels.
    :return: A nested dictionary with structure corresponding to the keys of the flattened
        dictionary passed in.
    """
    # Code adapted from link below.
    # https://www.geeksforgeeks.org/python-convert-flattened-dictionary-into-nested-dictionary
    def split_record(key, value, out):
        """Helper function to split up key."""
        # splitting keys in dict through recursive calls.
        key, *rest = key.split(separator, 1)
        if rest:
            split_record(rest[0], value, out.setdefault(key, {}))
        else:
            out[key] = value

    nested_dict: Dict[str, Any] = {}
    for k, v in flat_dict.items():
        # for each key call split_record which splits keys to form a recursively nested dictionary.
        split_record(k, v, nested_dict)
    return nested_dict


# TODO: Delete this function when TF issue 37252 is complete.
def monkey_patch_tf_get_seed(seed: int, default_op_seed: int = 1923746) -> None:
    """
    Monkey patching tensorflow.random.get_seed to avoid the increasing memory usage arising from
    repeated random sampling from tensorflow distributions.

    This code is taken from https://github.com/lerobitaille/tf-issue-36164-workaround which remedies
    issue 36164 (https://github.com/tensorflow/tensorflow/issues/36164).

    We have raised our own clearer and concise issue which should be the point at which should be
    the reference point for this memory leak: https://github.com/tensorflow/tensorflow/issues/37252

    :param seed: Seed to set as the TensorFlow global seed.
    :param default_op_seed: Default seed for any random operations if required.
    """
    warn("WARNING: Patching native TensorFlow functionality to avoid memory leak when setting "
         "a random seed.")
    warn("WARNING: Patch required due to TensorFlow issue 37252. "
         "Check if the issue is resolved at "
         "https://github.com/tensorflow/tensorflow/issues/37252")
    # Lazy imports to show which imports to remove once the issue is resolved and to avoid wider
    # usage of monkey patching and usage of the TensorFlow back end which involves imports the
    # linter does not like.
    # pylint: disable=no-name-in-module,import-error
    from tensorflow.python.eager import context
    from tensorflow.python import pywrap_tensorflow
    from tensorflow.python.framework import random_seed
    # Remove gorilla dependency completely when issue fixed. (Remove from requirements.txt)
    import gorilla

    def better_get_seed(global_seed, op_seed):
        if op_seed is not None:
            return global_seed, op_seed
        else:
            return global_seed, default_op_seed

    # Monkey Patch get_seed.
    def func(op_seed):
        better_get_seed(seed, op_seed)
    settings = gorilla.Settings(allow_hit=True, store_hit=True)
    patch = gorilla.Patch(random_seed, 'get_seed', func, settings=settings)
    gorilla.apply(patch)

    # Also clear the kernel cache, to reset any existing seeds
    # pylint: disable=protected-access
    _context = context.context()
    if _context._context_handle is not None:
        pywrap_tensorflow.TFE_ContextClearCaches(_context._context_handle)
