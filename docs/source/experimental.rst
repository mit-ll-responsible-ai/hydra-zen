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

Launching Hydra Job
====================

Using a defined task function and a configuration (we will focus on Structured Configs [1]_), 
a Hydra application is defined by decorating the task function with ``@hydra.main`` with a configuration name.

.. code-block:: python
   :caption: my_app.py

    from dataclasses import dataclass

    import hydra
    from hydra.core.config_store import ConfigStore

    @dataclass
    class MyExperiment:
        foo: str = "hello"
        bar: str = "world

    cs = ConfigStore.instance()
    # Registering the Config class with the name 'config'.
    cs.store(name="config", node=MyExperiment)

    @hydra.main(config_path=None, config_name="config")
    def task_function(cfg: MyExperiment) -> None:
        print(cfg.foo + ' ' + cfg.bar)

    if __name__ == "__main__":
        task_function()

Using the Hydra Command Line Interface (CLI), the application can be run by::

    $  python -m my_app.task_function bar=mom
    hello mom

hydra-zen provides functionality to execute Hydra for single or multirun mode in an interactive environment such as the Jupyter Notebook
by providing two experimental functions, ``hydra_run`` and ``hydra_multirun``, that mimic the behavior of Hydra's CLI. To demonstrate this, 
first convert ``my_app.py` to a hydra-zen application:

.. code-block:: python
   :caption: my_app_zen.py

    from hydra_zen import builds
    from hydra.core.config_store import ConfigStore

    MyExperiment = builds(foo = "hello", bar = "world")

    cs = ConfigStore.instance()
    # Registering the Config class with the name 'config'.
    cs.store(name="config", node=MyExperiment)

    def task_function(cfg: MyExperiment) -> None:
        print(cfg.foo + ' ' + cfg.bar)

Notice we no longer need the decorator for the task function.  Now, using ``hydra-run`` we can run our application in an interactive environment::

  >>> from my_app_zen import MyExperiment, task_function
  >>> from hydra_zen.experimental import hydra_run
  >>> job = hydra_run(MyExperiment, task_function, overrides=["bar=mom"])
  hello mom

In addition to running the experiments, we also have the return object. For the above application, the return object is a 
Hydra ``JobReturn`` object with the following attributes:

  - overrides: From `overrides` input
  - return_value: The return value of the task function
  - cfg: The configuration object sent to the task function
  - hydra_cfg: The hydra configuration object
  - working_dir: The experiment working directory
  - task_name: The task name of the Hydra job

To utilize Hydra Multirun and sweep over parameters, the Hydra CLI::

   $  python -m my_app.task_function bar=mom,dad --multirun
   hello mom
   hello dad

becomes::

   >>> from my_app_zen import MyExperiment, task_function
   >>> from hydra_zen.experimental import hydra_multirun
   >>> job = hydra_multirun(config, task_function, overrides=["bar=mom,dad"])
   hello mom
   hello dad

Additionally, since these functions end up executing Hydra we get all the benefits of Hydra's job configuration and logging. 
See Configuring Hydra [2]_ for more details on customizing Hydra.

Examples: hydra_run
*******************

The ``task_function`` is any function that can take a configuration object as input.  The simplest example is to just use ``instantiate``:

.. code:: python

    >>> from hydra_zen import instantiate, builds
    >>> from hydra_zen.experimental import hydra_run
    >>> job = hydra_run(builds(dict, a=1, b=1), task_function=instantiate)
    >>> job.return_value
    {'a': 1, 'b': 1}

Now lets define a more complex ``task_function``:

.. code:: python

    >>> from hydra_zen.experimental import hydra_run
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=10)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> job = hydra_run(cfg, task_function)
    >>> job.return_value
    100

An example using PyTorch:

.. code:: python

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, hydra_partial=True)
    >>> ModelConfig = builds(Linear, in_features=1, out_features=1)
    >>> cfg = dict(optim=AdamConfig, model=ModelConfig)
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
************************

Launch a Hydra multi-run [3]_ job by sweeping over configuration parameters:

.. code:: python

    >>> from hydra_zen import builds, instantiate
    >>> from hydra_zen.experimental import hydra_multirun
    >>> job = hydra_multirun(
    ...     builds(dict, a=1, b=1),
    ...     task_function=instantiate,
    ...     overrides=["a=1,2"],
    ... )
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]

Using a more complex ``task_function``:

.. code:: python

    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> jobs = hydra_multirun(cfg, task_function, overrides=["x=range(-2,3)"])
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]

An example ``task_function`` using PyTorch:

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

Now, evaluate the function for different learning rates:

.. code:: python

    >>> jobs = hydra_multirun(cfg, task_function, overrides=["optim.lr=0.1,1.0"])
    >>> [j.return_value for j in jobs[0]]
    [0.3054758310317993, 0.28910207748413086]


References
----------
.. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro
.. [2] https://hydra.cc/docs/configure_hydra/intro
.. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
