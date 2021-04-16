# hydra-zen

[![Automated tests status](https://github.com/mitll-SAFERai/hydra-zen/workflows/Tests/badge.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
[![Test coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
![Python version support](https://img.shields.io/badge/python-3.6%20&#8208;%203.9-blue.svg)

hydra-zen helps you configure your project using the power of [Hydra](https://github.com/facebookresearch/hydra), while enjoying the [Zen of Python](https://www.python.org/dev/peps/pep-0020/)!

It provides simple, Hydra-compatible tools that enable Python-centric workflows for designing and configuring large-scale
projects, such as machine learning experiments.
Configure and run your applications without leaving Python!

hydra-zen offers:
  - Functions for dynamically and ergonomically creating [structured configs](https://hydra.cc/docs/next/tutorials/structured_config/schema/) 
  that can be used to **fully or partially instantiate** objects in your application, using both user-specified and auto-populated configuration values.
  - The ability to launch hydra jobs, complete with parameter sweeps and multi-run configurations, from within a notebook or any
  other Python runtime.
  - Incisive type annotations that provide enriched context to IDEs, type checkers, and other tooling, about your project's
  configurations.
  - Runtime validation of configurations to catch mistakes before your application launches.

## Installation
`hydra-zen` is light weight: its dependencies are `hydra` and `typing-extensions`

```shell script
pip install hydra-zen
```

## Basic Usage

Let's use hydra-zen to configure an "experiment" that measures the impact of [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) when performing [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
down a 2D parabolic surface. 

```python
from dataclasses import dataclass
from typing import Any, Tuple

from torch.optim import SGD

from hydra_zen import builds, just

# defines our surface that we will be descending
def parabaloid(x, y):
    return 0.1 * x ** 2 + 0.2 * y ** 2

# defines the configuration for our experiment
@dataclass
class ExpConfig:
    starting_xy: Tuple[float, float] = (-1.5, 0.5)
    num_steps: int = 20
    optim: Any = builds(SGD, lr=0.3, momentum=0.0, hydra_partial=True)
    landscape_fn: Any = just(parabaloid)
```

Each `builds(<target>, ...)` and `just(<target>)` call creates a dataclass, which serves as a structured config for that `<target>`.
We can use `builds` configure an object to be either fully or partially instantiated using a combination of
user-provided and default values, and use `just` to have an object be returned without calling or instantiating it.

The resulting dataclass represents our experiment's configuration, from which we can adjust the starting condition, the number of
steps taken, the optimization method used, and even the landscape to be used. It will configure
the following function:

```python
import torch as tr
import numpy as np


def gradient_descent(*, starting_xy, optim, num_steps, landscape_fn):
    """Performs gradient descent down `landscape_fn`, and returns a trajectory
    of x,y values"""
    xy = tr.tensor(starting_xy, requires_grad=True)
    trajectory = [xy.detach().clone().numpy()]

    optim = optim([xy])

    for i in range(num_steps):
        z = landscape_fn(*xy)
        optim.zero_grad()
        z.backward()
        optim.step()
        trajectory.append(xy.detach().clone().numpy())
    return np.stack(trajectory)
```

Let's launch multiple hydra jobs, each configured to perform gradient descent with a different momentum value

```python
>>> from hydra_zen import instantiate  # annotated alias of hydra.utils.instantiate
>>> from hydra_zen.experimental import hydra_launch
 
>>> jobs = hydra_launch(
...     ExpConfig,
...     task_function=lambda cfg: gradient_descent(**instantiate(cfg)),
...     multirun_overrides=["optim.momentum=0.0,0.2,0.4,0.6,0.8,1.0"],
... )
[2021-04-15 21:49:40,635][HYDRA] Launching 6 jobs locally
[2021-04-15 21:49:40,635][HYDRA] 	#0 : optim.momentum=0.0
[2021-04-15 21:49:40,723][HYDRA] 	#1 : optim.momentum=0.2
[2021-04-15 21:49:40,804][HYDRA] 	#2 : optim.momentum=0.4
[2021-04-15 21:49:40,889][HYDRA] 	#3 : optim.momentum=0.6
[2021-04-15 21:49:40,975][HYDRA] 	#4 : optim.momentum=0.8
[2021-04-15 21:49:41,060][HYDRA] 	#5 : optim.momentum=1.0
```

And [plot](https://gist.github.com/rsokl/c7e2ed1aab02b35208bb5b4c8051a931) the results (omitting the plot-function definition for the
sake of legibility):

```python
>>> plot_jobs(jobs, fn=parabaloid)
```
![image](https://user-images.githubusercontent.com/29104956/114961883-b0b56580-9e37-11eb-9de1-87c8efc1780c.png)



`ExpConfig` is a standard structured config, thus it can used directly by Hydra to configure and run our application from the commandline.
Hydra can also serialize it to a yaml configuration file

```python
>>> from hydra_zen import to_yaml  # an alias for OmegaConf.to_yaml
>>> print(to_yaml(ExpConfig))
starting_xy:
- -1.5
- 0.5
num_steps: 20
optim:
  _target_: hydra_zen.funcs.partial
  _partial_target_:
    _target_: hydra_zen.funcs.identity
    obj: ${hydra_zen_get_obj:torch.optim.sgd.SGD}
  _recursive_: true
  _convert_: none
  lr: 0.3
  momentum: 0.0
landscape_fn:
  _target_: hydra_zen.funcs.identity
  obj: ${hydra_zen_get_obj:__main__.parabaloid}
```
