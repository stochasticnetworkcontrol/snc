import json
from typing import Dict, Optional

import numpy as np
from src.snc.environments import SCENARIO_CONSTRUCTORS


def load_env_params(json_file_name):
    """
    Load environment parameters from json file.

    :param json_file_name: Absolute path to the json file.
    :return: Dictionary with environment parameters.
    """
    with open(json_file_name) as json_file:
        env_param = json.load(json_file)

    for key, val in env_param.items():
        if isinstance(val, list):
            env_param[key] = np.array(val)
    return env_param


def load_env(env_name: str, env_param: Optional[Dict] = None):
    """
    Load environment object given by `env_name`, and with parameters given by `env_param`.

    :param env_name: Name environment example as defined in scenarios.SCENARIO_CONSTRUCTORS list.
    :param env_param: Dictionary with environment parameters.
    :return CRW environment object.
    """
    if env_param is not None:
        return SCENARIO_CONSTRUCTORS[env_name](**env_param)
    else:
        return SCENARIO_CONSTRUCTORS[env_name]()
