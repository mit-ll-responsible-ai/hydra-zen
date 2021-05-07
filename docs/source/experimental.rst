******************************
Running Experiments with Hydra
******************************

API Reference
====================

.. currentmodule:: hydra_zen.experimental

.. autosummary::
   :toctree: generated/

   hydra_run
   hydra_multirun

Launching Hydra Jobs
====================

Both ``hydra_run`` and ``hydra_multirun`` mimic the functionality of Hydra's CLI for consumption in an interactive 
environment such as the Jupyter Notebook.  Similar to how Hydra CLI works, the keyword argument, `overrides`, is a 
string list of configuration values to use for a given experiment run.  For example, the Hydra CLI provided by::

   $ python -m job.task_function job/group=group_name job.group.param=1

would be::

   >>> job = hydra_run(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

For ``hydra_multirun``, the Hydra CLI provided by::

   $ python -m job.task_function job/group=group_name job.group.param=1 --multirun

would be::

   >>> job = hydra_multirun(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

and to sweep over parameters using the Hydra CLI::

   $ python -m job.task_function job/group=group_name job.group.param=1,2,3 --multirun

would become::

   >>> job = hydra_multirun(config, task_function, overrides=["job/group=group_name", "job.group.param=1,2,3"])

Additionally, since these functions end up executing Hydra we get all the benefits of Hydra's job configuration and logging
such as the creation of a experiment working directory.  See Configuring Hydra [2]_ for more details on customizing Hydra.

Examples: hydra_run
====================

Launch a Hydra job defined by `task_function` using the configuration provided in `config`.

.. code:: python

    >>> from hydra_zen import instantiate, builds
    >>> from hydra_zen.experimental import hydra_run
    >>> job = hydra_run(builds(dict, a=1, b=1), task_function=instantiate)
    >>> job.return_value
    {'a': 1, 'b': 1}

Using a more complex task function:

.. code:: python

    >>> from hydra_zen.experimental import hydra_run
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=10)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

Launch a job to evaluate the function using the given configuration:

.. code:: python

    >>> job = hydra_run(cfg, task_function)
    >>> job.return_value
    100

An example using PyTorch:

.. code:: python

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, hydra_partial=True)
    >>> ModelConfig = builds(Linear, in_features=1, out_features=1)
    >>> cfg = dict(optim=AdamConfig(), model=ModelConfig())
    >>> def task_function(cfg):
    ...     cfg = instantiate(cfg)
    ...     optim = cfg.optim(model.parameters())
    ...     loss = cfg.model(torch.ones(1)).mean()
    ...     optim.zero_grad()
    ...     loss.backward()
    ...     optim.step()
    ...     return loss.item()
    >>> jobs = hydra_run(cfg, task_function, overrides=["optim.lr=0.1"])
    >>> j.return_value
    0.3054758310317993

Examples: hydra_multirun
========================

Launch a Hydra multi-run ([3]_) job defined by `task_function` using the configurationprovided in `config`.

.. code:: python

    >>> job = hydra_multirun(
    ...     builds(dict, a=1, b=1),
    ...     task_function=instantiate,
    ...     overrides=["a=1,2"],
    ... )
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]

Using a more complex `task_function`

.. code:: python

    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

Launch a multi-run over a list of different `x` values using Hydra's override syntax `range`:

.. code:: python

    >>> jobs = hydra_multirun(cfg, task_function, overrides=["x=range(-2,3)"])
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]

An example using PyTorch

.. code:: python

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, hydra_partial=True)
    >>> ModelConfig = builds(Linear, in_features=1, out_features=1)
    >>> cfg = dict(optim=AdamConfig(), model=ModelConfig())
    >>> def task_function(cfg):
    ...     cfg = instantiate(cfg)
    ...     optim = cfg.optim(model.parameters())
    ...     loss = cfg.model(torch.ones(1)).mean()
    ...     optim.zero_grad()
    ...     loss.backward()
    ...     optim.step()
    ...     return loss.item()

Evaluate the function for different learning rates

.. code:: python

    >>> jobs = hydra_multirun(cfg, task_function, overrides=["optim.lr=0.1,1.0"])
    >>> [j.return_value for j in jobs[0]]
    [0.3054758310317993, 0.28910207748413086]

References
----------
.. [1] https://hydra.cc/docs/advanced/override_grammar/basic
.. [2] https://hydra.cc/docs/configure_hydra/intro
.. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
