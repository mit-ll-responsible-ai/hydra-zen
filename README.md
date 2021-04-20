# hydra-zen

[![Automated tests status](https://github.com/mitll-SAFERai/hydra-zen/workflows/Tests/badge.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
[![Test coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/mitll-SAFERai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
![Python version support](https://img.shields.io/badge/python-3.6%20&#8208;%203.9-blue.svg)

hydra-zen helps you configure your project using the power of [Hydra](https://github.com/facebookresearch/hydra), while enjoying the [Zen of Python](https://www.python.org/dev/peps/pep-0020/)!

It provides simple, Hydra-compatible tools that enable Python-centric workflows for designing and configuring large-scale
projects such as machine learning experiments.
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
down a 2D parabolic surface. The following function uses PyTorch to perform gradient descent (via a user-specified optimizer)
down a given "landscape" function.



```python
# Setting up our code..
# This does not involve hydra-zen in any way.
import torch
import numpy

# This is the function that we will run under various configurations 
# in order to perform our analysis.
def gradient_descent(*, starting_xy, optim, num_steps, landscape_fn):
    """
    Parameters
    ----------
    starting_xy : Tuple[float, float]
    optim : Type[torch.optim.Optimizer]
    num_steps : int
    landscape_fn : (x: Tensor, y: Tensor) -> Tensor
    
    Returns
    -------
    xy_trajectory: ndarray, shape-(num_steps + 1, 2)
    """
    xy = torch.tensor(starting_xy, requires_grad=True)
    trajectory = [xy.detach().clone().numpy()]
    
    # `optim` needs to be instantiated with 
    # the tensor parameter(s) to be updated
    optim : torch.optim.Optimizer = optim([xy])

    for i in range(num_steps):
        z = landscape_fn(*xy)
        optim.zero_grad()
        z.backward()
        optim.step()
        trajectory.append(xy.detach().clone().numpy())
    return numpy.stack(trajectory)
```


To configure this function we'll define a dataclass that Hydra recognizes as a "structured config".
hydra-zen makes short work of generating a configuration that can "build" `gradient_descent` along with
the various Python objects that we want to pass to it.

```python
# Using hydra-zen to configure our code

from torch.optim import SGD

from hydra_zen import builds, just

# defines our surface that we will be descending
def parabaloid(x, y):
    return 0.1 * x ** 2 + 0.2 * y ** 2

# Defines dataclasses that configure `gradient_descent`
# and its parameters. Invalid parameter names will be caught here.
ConfigGradDesc = builds(
    gradient_descent,
    optim=builds(SGD, lr=0.3, momentum=0.0, hydra_partial=True),
    landscape_fn=just(parabaloid),
    starting_xy=(-1.5, 0.5),
    num_steps=20,
)

# `hydra_zen.typing` provides: `Builds`, `PartialBuilds`, and `Just` 
# 
# ConfigGradDesc              : Type[Builds[Type[gradient_descent]]]
# ConfigGradDesc.optim        : Type[PartialBuilds[Type[SGD]]]
# ConfigGradDesc.landscape_fn : Type[Just[Type[parabaloid]]]
# ConfigGradDesc.starting_xy  : Tuple[int, int]
# ConfigGradDesc.num_steps    : int
```

Each `builds(<target>, ...)` and `just(<target>)` call creates a dataclass that configures the target object.
`builds` can both auto-populate the configuration based on the target's signature and incorporate user-specified
defaults.
Thus `ConfigGradDesc` is simply a dataclass, and its parameters – even nested ones – can be accessed and modified in 
an intuitive way:

```python
# Examining the structured config for `gradient_descent`
>>> ConfigGradDesc
types.Builds_gradient_descent

>>> ConfigGradDesc.num_steps
20
>>> ConfigGradDesc.starting_xy
(-1.5, 0.5)

>>> ConfigGradDesc.optim
types.PartialBuilds_SGD
>>> ConfigGradDesc.optim.momentum
0.0
>>> ConfigGradDesc.optim.lr
0.3

>>> ConfigGradDesc.landscape_fn.path
'__main__.parabaloid'
```

Hydra provides us with a powerful means of recursively instantiating the objects that are
configured by these nested dataclasses, and thus call `gradient_descent` with these particular
configured values:

```python
# Instantiating our config and running `gradient_descent`
>>> from hydra_zen import instantiate # annotated alias of hydra.utils.instantiate
>>> instantiate(ConfigGradDesc)  # returns shape-(N, 2) array of xy values
array([[-1.5       ,  0.5       ],
       [-1.41      ,  0.44      ],
       ...
       [-0.43515927,  0.03878139]], dtype=float32)
```

Suppose that we want to run `gradient_descent` using the `SGD` optimizer configured with different momentum
values. We don't need to write boiler plate code to expose this particular parameter of this particular object in order
to adjust its value; Hydra makes it easy to override any of the above configured values and to recursively instantiate the
objects in our configuration with these values.

To demonstrate this, we'll use hydra-zen to launch multiple jobs from a Python console (or notebook) and configure each one to perform 
gradient descent with a different SGD-momentum value.

```python
# Running `gradient_descent` using multiple SGD-momentum values
>>> from hydra_zen.experimental import hydra_launch

>>> jobs = hydra_launch(
...     ConfigGradDesc,
...     task_function=instantiate,
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
(omitting the [plot-function's definition](https://gist.github.com/rsokl/c7e2ed1aab02b35208bb5b4c8051a931) for the sake of legibility):

```python
>>> func = instantiate(ConfigGradDesc.landscape_fn)  # Type[Just[Type[parabaloid]]] -> parabaloid
>>> plot_jobs(jobs, fn=func)
```
![image](https://user-images.githubusercontent.com/29104956/114961883-b0b56580-9e37-11eb-9de1-87c8efc1780c.png)

See that we could have configured _any_ aspect of this run in the same way. The learning rate, the number
of steps taken, the optimizer used (along with _its_ parameters), and even the landscape-function traversed can all be re-configured
and run in various combinations through this same succinct and explicit interface.
We could also dump `ConfigGradDesc` to a yaml configuration file and use that to configure our application.
This is the combined power of Hydra and the zen of Python in full effect!


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
