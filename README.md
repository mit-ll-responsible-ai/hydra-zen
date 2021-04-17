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
  that can be used to **fully or partially instantiate – or retrieve without instantiation –** objects in your application, using both 
  user-specified and auto-populated values to configure them.
  - The ability to launch hydra jobs, complete with parameter sweeps and multi-run configurations, from within a notebook or any
  other Python environment.
  - Incisive type annotations that provide enriched context to IDEs, type checkers, and other tooling, about your project's
  configurations.
  - Runtime validation of configurations to catch mistakes before your application launches.
  - Equal support for both object-oriented libraries (e.g. `torch.nn`) and functional ones (e.g. `jax` and `numpy`) 

## Installation
`hydra-zen` is light weight: its dependencies are `hydra-core` and `typing-extensions`

```shell script
pip install hydra-zen
```

## Brief Motivation

As your project grows in size, the process of configuring your experiments and applications can become 
highly cumbersome and end up a source of major technical debt. 
Extensive boiler plate code and repositories of generated static configurations are things to be avoided.
The tools supplied by hydra-zen help to keep your configuration process sleek and easy to reason about. 
Ultimately, hydra-zen promotes Python-centric workflows that are configurable, repeatable, and scalable.

## Basic Usage

Let's use hydra-zen to configure an "experiment" that measures the impact of [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) when performing [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
down a 2D parabolic surface. 

The following function is a toy example that uses PyTorch to perform gradient descent down a surface (a.k.a "landscape").
This does not involve hydra-zen, but we will use hydra-zen to configure the ways in which we run this function.


```python
import torch
import numpy

# This is the function that we will run under various configurations 
# in order to perform our analysis. 
# This code does not involve hydra-zen in any way.
def gradient_descent(*, starting_xy, optim, num_steps, landscape_fn):
    """Performs gradient descent down `landscape_fn`, and returns a trajectory
    of x,y values.
    
    Parameters
    ----------
    starting_xy : Tuple[float, float]
    optim : torch.optim.Optimizer
    num_steps : int
    landscape_fn : (x: Tensor, y: Tensor) -> Tensor
    
    Returns
    -------
    xy_trajectory: ndarray, shape-(num_steps + 1, 2)
    """
    xy = torch.tensor(starting_xy, requires_grad=True)
    trajectory = [xy.detach().clone().numpy()]
    
    # `optim` is only partially instantiated and needs
    # to be passed the tensor parameter  
    optim = optim([xy])

    for i in range(num_steps):
        z = landscape_fn(*xy)
        optim.zero_grad()
        z.backward()
        optim.step()
        trajectory.append(xy.detach().clone().numpy())
    return numpy.stack(trajectory)
```


To configure this function in a simple but expressive way, we define a dataclass that Hydra 
recognizes as a "structured config".
hydra-zen makes short work of generating configurations for the various Python objects
that we need in our function: our optimizer class and our "landscape" function.

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
    # `optim` is configured as a partial-instantiation of SGD.
    # (we wont fully instantiate it until we have access to the
    # tensor that we are optimizing in our function). 
    optim: Any = builds(SGD, lr=0.3, momentum=0.0, hydra_partial=True)  # type: Type[PartialBuilds[Type[SGD]]]
    
    # `landscape_fn` is configured to be 'just' the un-instantiated the 
    # `parabaloid` function 
    landscape_fn: Any = just(parabaloid)  # type: Type[Just[(Any, Any) -> Any]]
    
    starting_xy: Tuple[float, float] = (-1.5, 0.5)
    num_steps: int = 20
```

Each `builds(<target>, ...)` and `just(<target>)` call creates a dataclass that configures the target object.
`builds` can both auto-populate the configuration based on the target's signature and incorporate user-specified
defaults.
Thus these parameters - even nested ones – can be accessed and modified in an intuitive way:

```python
>>> ExpConfig.num_steps
20 
>>> ExpConfig.optim.momentum
0.0
>>> ExpConfig.landscape_fn.path
'__main__.parabaloid'
```

Let's see what this configuration looks like as a yaml file (which Hydra can use to configure
and run our function from the commandline):

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
    _target_: hydra_zen.funcs.get_obj
    path: torch.optim.sgd.SGD
  _recursive_: true
  _convert_: none
  lr: 0.3
  momentum: 0.0
landscape_fn:
  _target_: hydra_zen.funcs.get_obj
  path: __main__.parabaloid
```

Suppose that we want to run `gradient_descent` using the `SGD` optimizer configured with different momentum
values. We don't need to write boiler plate code to expose this particular parameter of this particular object in order
to adjust its value; Hydra makes it easy to override any of the above configured values.
To see this, we'll launch multiple hydra jobs from a Python console (or notebook) and configure each one to perform 
gradient descent with a different SDG-momentum value.

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

Let's plot the trajectories produced by these jobs 
(omitting the [plot-function definition](https://gist.github.com/rsokl/c7e2ed1aab02b35208bb5b4c8051a931) for the sake of legibility):

```python
>>> plot_jobs(jobs, fn=parabaloid)
```
![image](https://user-images.githubusercontent.com/29104956/114961883-b0b56580-9e37-11eb-9de1-87c8efc1780c.png)

See that we could have configured _any_ aspect of this run in the same way. The learning rate, the number
of steps taken, the optimizer used (along with _its_ parameters), and even the landscape-function traversed can all be re-configured
and run in various combinations through this same succinct and explicit interface.
This is the power of Hydra and the zen of Python!


## Disclaimer
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and 
Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or 
recommendations expressed in this material are those of the author(s) and do not necessarily reflect 
the views of the Under Secretary of Defense for Research and Engineering.

© 2021 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 
7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are 
defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other 
than as specifically authorized by the U.S. Government may violate any copyrights that exist in 
this work.
