# The SNC toolbox
This toolbox provides a general simulation environment, with multiple examples, and a 
number of sequential decision-making agents for stochastic network control (SNC) problems. 

The Stochastic Network Control (SNC) toolbox was initially developed at 
[Secondmind](http://secondmind.ai/ "Secondmind") by 
[Sergio Valcarcel Macua](https://github.com/sergiovalmac), 
[Egor Tiavlovsky Vasilevitch](https://github.com/tiavlovsky), 
[Sofia Ceppi](https://github.com/sofiaceppi), 
[Ian Davies](https://github.com/IanRDavies), 
[Eric Hambro](https://github.com/condnsdmatters), 
[Arnaud Cadas](https://github.com/ArnaudCadas), 
[Daniel McNamee](https://github.com/dmcnamee), 
[Alexis Boukouvalas](https://github.com/alexisboukouvalas), 
[Enrique Munoz de Cote](https://github.com/enriquedecote) 
and 
[Sean Meyn](https://github.com/supermeyn).

A paper with the description of the simulator environment and the Hedgehog agent will be available 
soon.

# What is Stochastic Network Control?
Stochastic Network Control (SNC) provides optimal control tools for analysing and optimising 
stochastic queuing networks. There are many complex and challenging optimisation problems arising 
from managing these networks, which are typically referred to as scheduling, routing, and inventory 
optimisation. For these problems, SNC allows one to minimise the overall long term cost while 
ensuring a target service quality level thus improving overall asset utilisation efficiency. 

We focus on heavily loaded networks. In this regime, the network operates at near-maximum capacity 
whereby most of the resources are spent in attending to the stream of new arrivals (orders, 
customers, jobs, cars, and so on). Therefore, there is little room for inefficiencies or exploration 
before the network becomes unstable. If the queues become full, it will take a significant amount of 
time to drain them, as the resources have little extra capacity available after attending to the 
incoming demand. A heavily loaded state might be the normal operating condition for some networks. 
Alternatively, this scenario can arise temporarily as a result of unexpected increases in jobs/items to 
be processed (such as is the case in the global supply chain due to the pandemic) or resource 
malfunctions (observed in natural ecosystems due to climate disruptions). Heavily loaded conditions 
are the most challenging for which other existing tools fail. 

# Simulation environment
The simulation environment allows simulations of a controlled random walk (CRW) over a queuing 
network. A queuing network is composed of queues, resources that serve those queues by routing the 
items to other queues, and constraints on both queues and resources. Both the arrivals of items to 
the network and the time taken to process items from the queues are stochastic processes.

The CRW is a general environment with a number of parameters, like the network topology, costs and 
demand rates. We have implemented multiple instances of the CRW, some of which correspond with 
standard problems in the queuing network literature and that are useful to illustrate different 
behaviours, like tendency to instability under myopic policies.

The simulation environment is compatible with [OpenAI Gym](https://github.com/openai/gym) and we 
have also included a wrapper for compatibility with 
[TensorFlow Agents](https://www.tensorflow.org/agents).

# Agents
The main contribution of this toolbox is a model-based agent coined _Hedgehog_, which is an 
extension of the seminal research on workload relaxations by Prof. Sean Meyn 
(see [[1]](http://www.meyn.ece.ufl.edu/archive/CTCNonline.pdf) and references therein). 
We also include a number of baselines.

  
[[1]](http://www.meyn.ece.ufl.edu/archive/CTCNonline.pdf) Meyn, S., 2008. Control techniques for 
complex networks. Cambridge University Press 
(available [online](http://www.meyn.ece.ufl.edu/archive/CTCNonline.pdf)).

## Hedgehog
Heavily loaded networks can be controlled at two timescales: 
* **Slow timescale:** Since the resources work at capacity, the network drains very slowly; and the 
  most loaded resources determine the minimum draining time of the network. 
  We call these resources bottlenecks.
* **Fast timescale:** If any resource idles, it will accumulate items in its queues, but it also 
  frees other resources from the stream of arrivals, reducing their instantaneous load and allowing 
  them to drain their queues very quickly.

Hedgehog exploits these two timescales to drain the most expensive queues very quickly, while 
ensuring drainage in the minimum time. Under some circumstances, we can prove that this is the 
behaviour of the optimal policy.

Main features of _Hedgehog_ are:
* **High performance.** Hedgehog has been shown to outperform baselines under transitory and 
  steady-state regimes. This is expected as the theory predicts that it achieves near optimal 
  performance for heavily loaded networks.
* **Wide applicability.** Hedgehog can be applied to a wide range of complex scenarios which may 
  vary according to network topologies, activities, constraints and statistical assumptions.
* **Reduced complexity.** Hedgehog relies on the so-called _workload relaxation_ 
  to reduce the dimensionality of the problem and avoid the combinatorial complexity of selecting 
  actions in these networks 
  (see Chapter 5 of [[1](http://www.meyn.ece.ufl.edu/archive/CTCNonline.pdf)]).
* **Explainability.** Hedgehog follows a policy-synthesis approach that responds to key actionable 
  information of the network, like resources having queues below some safety stock thresholds, 
  the need for hedging on costly resources, or the need for keeping bottlenecks active to avoid 
  increasing the draining time. These key factors naturally make the recommendations fully 
  explainable, enabling the people making consequential decisions to understand the assumptions made 
  by the algorithm, and the reasons supporting each suggested action.
* **Adaptability.** Although not implemented yet, we expect Hedgehog to be highly adaptable to 
  expected or unexpected disruptions, like preventive maintenance or breakdowns. The algorithm 
  relies on prior knowledge which is usually available for these networks, like the network 
  topology, and on some modest model estimation that characterises the external stochastic 
  processes. If the network topology changes (one resource becomes unavailable due to breakdown, or 
  a transportation route is collapsed), the information can be passed to Hedgehog, and it will adapt 
  its policy in the next time step. Strategies for dealing with drift on the external stochastic 
  processes include continuous estimation during operation or switching to a stable (but 
  conservative) policy while the statistics of the external stochastic processes are re-estimated, 
  before switching back to Hedgehog.

## Baselines
We include the following baselines:
* _MaxWeight_ and _BackPressure_: Family of control algorithms where the decision to operate any 
  activity is determined by considering the difference between weighted states of outflow and inflow 
  buffers. In the case of BackPressure, only relative buffer levels are considered; 
  whereas MaxWeight scales individual buffer levels by their corresponding costs
  (see, e.g., Sections 4.8 and 6.4 of [[1](http://www.meyn.ece.ufl.edu/archive/CTCNonline.pdf)]).
* Priority based heuristics: family of heuristics where the actions are chosen based on factors like 
  which is the queue with largest number of items, what is the resource with the lowest processing 
  rate, which queue has the highest cost, or any combination of the above.
* TensorFlow Agents: We include wrappers to be compatible with the model-free reinforcement learning 
  baselines implemented in [TensorFlow Agents](https://www.tensorflow.org/agents). TensorFlow Agents 
  includes implementations of some well known model-free reinforcement learning algorithms which can 
  be easily adapted to the SNC codebase. At the moment we have adapted REINFORCE and PPO agents.
* Environment specific agents: Rule-based heuristics specifically designed for some environments.
* Application specific agents: SNC can be applied to a wide range of applications. Some of them
  have their own family of application-specific baselines. For example, multi-echelon inventory 
  optimisation (MEIO) is an application for which there exist multiple algorithms to set the safety 
  stock levels at every stage of the supply chain based on the lead times and demand distributions.
  You can find an implementation of the Guaranteed Service Model (GSM) algorithm under `meio` 
  folder. However, a whole GSM SNC agent based on that is not ready yet.  

# Installation instructions
At present, the packaging of this repository as a python package is incomplete. Having a clone of 
this repository locally is necessary to utilise the full functionality.

SNC supports Python 3.7 onwards.

For Ubuntu 18.04 preliminary installation of GCC and the `libgmp3-dev` library is required for 
installing the `pycddlib` python package:
```bash
sudo apt-get install libgmp3-dev
sudo apt-get install python3.7-dev
```

To install this project in editable mode run the command below from the root directory of the 
`snc` repository:
```bash
pip install -r requirements.txt 
pip install -e .
```

# Running experiments
The main script to run experiments is [validation_script](src/snc/simulation/validation_script.py), 
which allows us to simulate any of the available agents, including trained reinforcement learning 
agents.

The validation script takes a whitespace separated list of agent names
in its `--agents` argument, a scenario name in its `--env_name` argument and environment parameters
in its `--env_param_overrides` (or `-ep` for short) argument. For any missing environment parameters 
the default value is used.

Example of json files for environments and agents are included in the folder 
`snc/simulation/json_examples_validation_script/`.

For example, the following command runs Hedgehog for 2000 timesteps in the `simple_reentrant_line` 
environment, with agent and environment parameters given by 
`/path_to_file/hedgehog.json` and `/path_to_file/simple_reentrant_line.json`, respectively, 
and with discount factor 0.99999:
```bash
python snc/simulation/validation_script.py -ns 2000 --discount_factor 0.99999 --seed 0 --agents "bs_hedgehog" -hhp "/path_to_file/hedgehog.json" --env_name "simple_reentrant_line" -ep "/path_to_file/simple_reentrant_line.json"
```
While the following command runs Hedgehog and MaxWeight sequentially in the same environment and 
with same parameters as above, and with additional parameters for the MaxWeight agent given by 
`/path_to_file/maxweight.json`:
```bash
python snc/simulation/validation_script.py -ns 2000 --discount_factor 0.99999 --seed 0 --agents "bs_hedgehog maxweight" -hhp "/path_to_file/hedgehog.json" -mwp "/path_to_file/maxweight.json" --env_name "simple_reentrant_line" -ep "/path_to_file/simple_reentrant_line.json"
```

### Running trained reinforcement learning agents

To load trained reinforcement learning agents pass a list of paths to the directories to the 
`--rl_checkpoints` argument of the [validation_script](src/snc/simulation/validation_script.py). These
directories should be in an order corresponding to the order of the agent list passed to `--agents`.
This also supports the running of multiple instances of the same RL agent with different saved
parameters (e.g. `--agents "ppo ppo" --rl_checkpoints "['weights_dir_1', 'weights_dir_2']`).
Note that when using the special case of `--agents all`, `validation_script.py` expects a list of
two directories (as above) to `--rl_checkpoints` and the directory for REINFORCE must precede the
one for PPO.


# Running the tests

These commands should be run from the root directory of this repository.

```bash
pip install -r requirements.txt
tox
```

# Building the docker image

This repo can also be built with the required dependencies in a docker image. This may help when 
running experiments in a cloud environment.
In order to build a docker image, run the following commands from the root directory of this 
repository.
The resulting image will contain all of the code in the `snc` directory of this repository and have 
all the dependencies installed.

```bash
docker build . -t <image identifier>
```

The docker image will be built according to the commands in `Dockerfile` in the root directory of
this repository. 

The important elements of the dockerfile are the `COPY` command(s) which copy code from the
repository into the docker image. These are required for the code to be used within the docker image.
Further to this, at the end of the dockerfile there may be an `ENTRYPOINT` which specifies
which script to run by default under the command `docker run` (`docker run` is the command that runs
the content of a docker image). This entry point can be overwritten
by passing a path to a new entry point (within the docker image) as an argument following the 
`--command` modifier. Note that the `--command` argument also takes the command to run the script
so a valid usage would be `--command "python script.py"`.

For this repository we do not specify an entry point so that the image is generic and an entry point
must be specified when running experiments. This means that we only maintain a single `Dockerfile`.

**Note:** The current `.dockerignore` file is set up to ignore all files and directory not
explicitly included. Files and directories can be included using `!<file or directory>`.


# Contributions
We welcome contributions. Contribution guidelines will be added soon. 
