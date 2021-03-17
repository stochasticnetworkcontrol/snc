## SNC Experiments

#### Preparing to Experiment
An overview of how to build a docker image for this repo and a summary docker usage is provided
in the [main README](../README.md) for this repository. Once you have
built a docker image test it locally and run a single experiment run before setting up multiple
experiments.

To run a docker image locally use `docker run -it <image identifier> <command>` where the command determines what to
run, for example `python /code/snc/experiment/rl/rl_experiment_script.py`. Running `bash` provides
an interactive shell which is useful for debugging any docker build issues.


#### Writing a JSON Config File
The experiments can be defined using a JSON config file. [`command_builder.py`](experiments_from_config/command_builder.py)
supports the definition of a series of distinct experiments as a list of configurations (for an example
see [`example_list_config.json`](experiments_from_config/json_examples/example_list_config.json)) or defining spaces of parameters to
sweep over running a new experiment for each combination of parameters (for an example see
[`example_sweep_config.json`](experiments_from_config/json_examples/example_sweep_config.json)). The type of experiment
specification is inferred, by `command_builder.py`, from whether the values for each environment name are lists (as for list
configs) or dictionaries (as for sweep configs).

The JSON parsing is handled by [`command_builder.py`](experiments_from_config/command_builder.py) which should be the first port of
call for debugging. Note that this means that the JSON parsing is specific to this repository.

Note that there may be issues when running with GPUs. Due to Kubernetes not using the latest Nvidia drivers there will be warnings when running
experiments with GPUs. These are unavoidable at the time of implementation but should not affect performance
for the codebase (as we do not use anything from `tf.experimental`). The key line to see in the logs
to ensure that GPU is being used is the below (or similar).
```
tensorflow/core/common_runtime/gpu/gpu_device.cc:1555 Found device 0 with properties: 
name: Tesla K80 computeCapability: 3.7
coreClock: 0.8235GHz coreCount: 13 deviceMemorySize: 11.17GiB deviceMemoryBandwidth: 223.96GiB/s
...
tensorflow/compiler/xla/service/service.cc:176   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
```  

Note: [command_builder.py](experiments_from_config/command_builder.py) is only able to handle one set of agent parameters at a time
and therefore where trying to build commands involving the [validation_script](../snc/simulation/validation_script.py)
we recommend running each agent in a separate run.

 ##### Experiment List Configuration
A JSON file defining a list of experiments will have the environment name  as defined and implemented in [`scenarios.py`](../snc/environments/scenarios.py)
as a first level key and then the value will be a list of parameter dictionaries, one for each experiment.
The named environments must be implemented in [`scenarios.py`](../snc/environments/scenarios.py).
A code snippit from [`example_list_config.json`](experiments_from_config/json_examples/example_list_config.json) is included
below for illustration.
```json
{
  "klimov_model":
    [
      {
        "lab_name": "klimov",
        "max_episode_length": 300,
        "eval_freq": 50,
        "save_freq": 25,
        "log_freq": 10,
        "num_iters": 10000,
        "env_param_overrides": {
          "alpha1": 0.95,
          "alpha2": 0.95,
          "alpha3": 0.95,
          "alpha4": 0.95,
          "mu1": 1.0,
          "mu2": 1.0,
          "mu3": 1.0,
          "mu4": 1.0,
          "cost_per_buffer": [[1],[1],[1],[1]]
        }
      },
      {
        "lab_name": "klimov",
        "max_episode_length": 300,
        "eval_freq": 50,
        "save_freq": 25,
        "log_freq": 10,
        "num_iters": 10000,
        "env_param_overrides": {
          "alpha1": 0.95,
          "alpha2": 0.95,
          "alpha3": 0.95,
          "alpha4": 0.95,
          "mu1": 1.0,
          "mu2": 1.0,
          "mu3": 1.0,
          "mu4": 1.0,
          "cost_per_buffer": [[1],[2],[4],[8]]
        }
      }
    ]
}
```
For the list configuration each parameter is provided with a corresponding value.
There is one parameter argument which does not take a single value. This is the special
case of the `env_param_overrides` argument. When passed to target python scripts, this is
a JSON string representing a dictionary of values to be used to override the environment's default values.
We automatically generate these strings from the dictionary of parameter-value pairs which is
passed as a value to the `env_param_overrides` key.

##### Parameter Sweep Configuration
When defining a set of parameter spaces to sweep over, in the JSON file the first level keys are the
environment names. Again these environments must be implemented in [`scenarios.py`](../snc/snc/environments/scenarios.py).
In the example below the environments being run are `single_server_queue` and `ksrs_network_model`
(taken from [`example_sweep_config.json`](experiments_from_config/json_examples/example_sweep_config.json)).

```json
{
  "single_server_queue": {
    "lab_name": "ssq",
    "repeats" : 2,
    "discount_factor": ["set", [0.9, 0.99]],
    "env_param_overrides": {
      "demand_rate_val": ["set",  [0.1, 0.25, 0.8, 0.95, 0.99]],
      "initial_state": ["set", [[[0]], [[100]]]]
    }
  },
  "ksrs_network_model": {
    "lab_name": "ksrs",
    "discount_factor": ["frange", [0.8, 1.0, 0.1]],
    "num_iters": ["set", [10, 100]],
    "env_param_overrides" : {
      "alpha1": ["irange", [6, 10, 1]],
      "job_conservation_flag": ["set", [true, false]]
    }
  }
}
```

For each environment a set of experiment parameters to iterate through is defined. These definitions
take the form of `parameter_name: sweep_space_specification`. The exceptions to this are the
`repeats` value, which determines how many times to repeat _each_ experiment for the given environment,
and `lab_name` which defines the lab in which experiments will be run (which essentially defines
the directory for saving results).
If no value is passed for `repeats` only one instance of each experiment is run.
If no value is passed for `lab_name` the lab name will default to `"default_lab"`. All repeats will
be run in the same lab.

The experiment set up currently supports three kinds of parameter 'space' to iterate through in experiments:
`set`, `frange` and `irange`. In this case a `set` is considered as a collection of discrete items which can be
iterated over. A `frange` considers a continuous space and takes a `start`, `stop` and `step` argument
and acts similarly to Python's built-in `range` class and an `irange` is an integer range which is
implemented as a python `range`. *Note:* The floating point range may suffer from
inaccuracies due to the numbers being floating point representations.

An example `set` can be seen in the case of `discount_factor` for `single_server_queue` in the example
above. In the case of `ksrs_network_model` an `frange` is used for `discount_factor` instead. An example
of the integer range (`irange`) would be the case of `alpha1` in the `ksrs_network_model` experiments.

There is one parameter argument which does not take a sweep space specification. This is the special
case of the `env_param_overrides` argument. When passed to target python scripts, this is
a JSON string representing a dictionary of values to be used to override the environment's default values.
We automatically generate these strings from the specifications laid out in the dictionary
which is passed as a value to the `env_param_overrides` key. Parameter sets for the environment parameter
overrides are then generated per experiment in the same way as the other arguments at the second level.

*Note:* Numpy arrays are not compatible with standard JSON representation and should therefore be written
as lists and will be cast to numpy arrays as appropriate in the python script which is ultimately
run in the experiment.

The space of parameters which is ultimately iterated over is given by the Cartesian product of the
spaces for each parameter within each environment. _*This means that the number of experiments being
run will face combinatorial explosion if the individual parameter search spaces are large and/or
numerous. One experiment is run for each combination of parameters possible in the specification for
each environment.*_