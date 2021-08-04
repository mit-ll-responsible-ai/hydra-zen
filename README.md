# hydra-zen

[![Automated tests status](https://github.com/mit-ll-responsible-ai/hydra-zen/workflows/Tests/badge.svg)](https://github.com/mit-ll-responsible-ai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
[![Test coverage](https://img.shields.io/badge/coverage-100%25-green.svg)](https://github.com/mit-ll-responsible-ai/hydra-zen/actions?query=workflow%3ATests+branch%3Amain)
[![Docs status](https://github.com/mit-ll-responsible-ai/hydra-zen/actions/workflows/publish_docs.yml/badge.svg)](https://mit-ll-responsible-ai.github.io/hydra-zen/)
![Python version support](https://img.shields.io/badge/python-3.6%20&#8208;%203.9-blue.svg)

hydra-zen helps you configure your project using the power of [Hydra](https://github.com/facebookresearch/hydra), while enjoying the [Zen of Python](https://www.python.org/dev/peps/pep-0020/)!

hydra-zen eliminates the boilerplate code that you write to configure, orchestrate, and organize the results of large-scale projects, such as machine learning experiments. It does so by providing Hydra-compatible tools that dynamically generate "structured configurations" of your code, and enables Python-centric workflows for running configured instances of your code.

hydra-zen offers:
  - Functions for automatically and dynamically generating [structured configs](https://hydra.cc/docs/next/tutorials/structured_config/schema/) that can be used to fully or partially instantiate objects in your application.
  - The ability to launch Hydra jobs, complete with parameter sweeps and multi-run configurations, from within a notebook or any other Python environment.
  - Incisive type annotations that provide enriched context about your project's configurations to IDEs, type checkers, and other tooling.
  - Runtime validation of configurations to catch mistakes before your application launches.
  - Equal support for both object-oriented libraries (e.g., `torch.nn`) and functional ones (e.g., `jax` and `numpy`).

These functions and capabilities can be used to great effect alongside [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) to design boilerplate-free machine learning projects!  

Check out [this example](https://mit-ll-responsible-ai.github.io/hydra-zen/pytorch_lightning_example.html) to see how hydra-zen and PyTorch-Lightning can be used to create a boilerplate-free ML extension.

[This repo](https://github.com/mit-ll-responsible-ai/hydra-zen-examples) contains some end-to-end ML projects that are configured using hydra-zen. This can provide guidance for how to organize a project to make use of hydra-zen.  

## Installation
`hydra-zen` is lightweight: its only dependencies are `hydra-core` and `typing-extensions`.

```shell script
pip install hydra-zen
```

## Diving into hydra-zen
### Building Basic Configs with hydra-zen

hydra-zen provides simple, but powerful, functions for creating rich configurations.
These structured configs can be used to launch Hydra jobs (from CLI or within Python).
They can also be serialized to yaml files.

This section presents the ABCs of using hydra-zen to configure your project.

`builds(<target>, ...)` creates a dataclass that tells Hydra how to "build" `<target>`
with both user-specified and auto-populated parameter values.

```python
>>> from hydra_zen import builds

# A configuration describing how to "build" a particular dictionary
>>> BuildsDict = builds(dict, hello=1, goodbye=None)

# signature: BuildsDict(hello: Any = 1, goodbye: Any = None)
>>> BuildsDict  # A class-object with the following configurable attrs...
types.Builds_dict

>>> BuildsDict._target_
'builtins.dict'

>>> BuildsDict.hello
1

>>> BuildsDict.goodbye
None

# overriding `goodbye` by making an instance of the dataclass
>>> BuildsDict(goodbye=2)
Builds_dict(_target_='builtins.dict', hello=1, goodbye=2)
```

Hydra's `instantiate` function is used to enact this build. This can be used in a recursive fashion:

```python
>>> from hydra_zen import instantiate  # annotated alias of hydra.utils.instantiate

>>> instantiate(BuildsDict)  # calls `dict(hello=1, goodbye=None)`
{'hello': 1, 'goodbye': None}

# recursively instantiating nested builds
>>> def square(x): return x ** 2
>>> instantiate(builds(square, builds(square, 2)))  # calls `square(square(2))`
16
```

The `just(<target>)` function creates a configuration that "just" returns the target (a non-literal Python object), without calling/initializing it.
`builds` will automatically apply `just` to created nested structured configs.

```python
>>> from hydra_zen import just

# configuration says: "just" return the square function
>>> JustSquare = just(square)
>>> instantiate(JustSquare)
<function __main__.square(x)>

# `builds` will automatically apply `just` to functions and classes
>>> NumberConf = builds(dict, initial_val=2., transform=square)
>>> NumberConf.transform
types.Just_square
>>> instantiate(NumberConf)  # calls `dict(number_type=int, transform=square)`
{'initial_val': 2.0, 'transform': <function __main__.square(x)>}
```

The dataclasses produced by `builds` and `just` are valid [structured configs](https://hydra.cc/docs/next/tutorials/structured_config/intro) for Hydra to use, thus they can be serialized to yaml configuration files, which can later be loaded and "instantiated" for reproducible results.

```python
>>> from hydra_zen import to_yaml  # alias of `omegaconf.OmegaCong.to_yaml`
>>> print(to_yaml(NumberConf))
_target_: builtins.dict
initial_val: 2.0
transform:
  _target_: hydra_zen.funcs.get_obj
  path: __main__.square
```

Don't let the simplicity of these functions deceive you!
There are many expressive and rich patterns to be leveraged here in addition to
functionality not yet discussed.

### A Simple Application of hydra-zen

It's time to see how we can use hydra-zen in an applied setting.

Let's use hydra-zen to both configure and run an "experiment" using Hydra, but without ever leaving our Python environment.
Our experiment will measure the impact of [momentum](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum) when performing [gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)
down a 2D parabolic surface. The following function uses PyTorch to perform gradient descent (via a user-specified optimizer)
down a given "landscape" function.


```python
# Setting up our code...
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
    optim: torch.optim.Optimizer = optim([xy])

    for i in range(num_steps):
        z = landscape_fn(*xy)
        optim.zero_grad()
        z.backward()
        optim.step()
        trajectory.append(xy.detach().clone().numpy())
    return numpy.stack(trajectory)
```


We will use `build` and `just` to configure `gradient_descent`
as well as the parameter values that we want to pass to it.
Note that we want our optimizer, `SGD`, to only be *partially* built by our config given that it is fully initialized within `gradient_descent`; `builds` makes simple work of this. 

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

    # Configured to only partially build `SGD`.
    # Not all of its required inputs are part of our configuration
    optim=builds(SGD, lr=0.3, momentum=0.0, hydra_partial=True),

    landscape_fn=parabaloid,
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

Keep in mind that `ConfigGradDesc` is simply a dataclass, and its parameters – even nested ones – can be accessed and modified in 
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

As we saw earlier, `instantiate` will recurse through our nested configurations
and "build" (or "just" return) our configured objects.
Thus it will call `gradient_descent` with its configured parameters

```python
# Instantiating our config and running `gradient_descent`
>>> from hydra_zen import instantiate
>>> instantiate(ConfigGradDesc)  # calls `gradient_descent(optim=partial(SGD, ...), ...)
array([[-1.5       ,  0.5       ],
       [-1.41      ,  0.44      ],
       ...
       [-0.43515927,  0.03878139]], dtype=float32)
```

Now suppose that we want to run `gradient_descent` multiple times – each run with the `SGD` optimizer configured with a different momentum value. 
Because we are using hydra-zen, we don't need to write boilerlate code to expose this particular parameter of this particular object in order to adjust its value.

To demonstrate this, we'll use hydra-zen to launch multiple jobs from a Python console (or notebook) using Hydra's default sweeper and launcher plugins, and configure each one to perform gradient descent with a different SGD-momentum value.

```python
# Running `gradient_descent` using multiple SGD-momentum values
>>> from hydra_zen.experimental import hydra_multirun

# Returns a List of Hydra's JobReturn objects for each experiment
>>> jobs = hydra_multirun(
...     ConfigGradDesc,
...     task_function=instantiate,
...     overrides=["hydra/sweeper=basic", "optim.momentum=range(0.0,1.2,0.2)"],
... )
[2021-04-15 21:49:40,635][HYDRA] Launching 6 jobs locally
[2021-04-15 21:49:40,635][HYDRA] 	#0 : optim.momentum=0.0
[2021-04-15 21:49:40,723][HYDRA] 	#1 : optim.momentum=0.2
[2021-04-15 21:49:40,804][HYDRA] 	#2 : optim.momentum=0.4
[2021-04-15 21:49:40,889][HYDRA] 	#3 : optim.momentum=0.6
[2021-04-15 21:49:40,975][HYDRA] 	#4 : optim.momentum=0.8
[2021-04-15 21:49:41,060][HYDRA] 	#5 : optim.momentum=1.0
>>> jobs[0][1].return_value  # corresponds to momentum=0.2
array([[-1.5       ,  0.5       ],
       [-1.41      ,  0.44      ],
       ...
       [-0.43515927,  0.03878139]], dtype=float32)
```
In this example we used Hydra's default launcher, but its various plugins – launchers (e.g., the [SLURM-based launcher](https://hydra.cc/docs/next/plugins/submitit_launcher)) and parameter sweepers – can be used here.

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

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

A portion of this research was sponsored by the United States Air Force Research Laboratory and the United States Air Force Artificial Intelligence Accelerator and was accomplished under Cooperative Agreement Number FA8750-19-2-1000. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the United States Air Force or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.

© 2021 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

