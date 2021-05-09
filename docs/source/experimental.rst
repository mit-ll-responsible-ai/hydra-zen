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
        bar: str = "world"

    cs = ConfigStore.instance()
    # Registering the Config class with the name 'config'.
    cs.store(name="config", node=MyExperiment)

    @hydra.main(config_path=None, config_name="config")
    def task_function(cfg: MyExperiment) -> None:
        print(cfg.foo + ' ' + cfg.bar)

    if __name__ == "__main__":
        task_function()

Using the Hydra Command Line Interface (CLI), the application can be run by::

    $  python my_app.py bar=mom
    hello mom

hydra-zen provides functionality to execute Hydra for single or multirun mode in an interactive environment such as the Jupyter Notebook
by providing two experimental functions, ``hydra_run`` and ``hydra_multirun``, that mimic the behavior of Hydra's CLI.
Using ``hydra-run`` we can run our application in an interactive environment::

  >>> from my_app_zen import MyExperiment, task_function
  >>> from hydra_zen.experimental import hydra_run
  >>> job = hydra_run(MyExperiment, task_function, overrides=["bar=mom"])
  hello mom

In addition to running the experiments we also have the return object. For the above application the return object is a
Hydra ``JobReturn`` object with the following attributes:

  - overrides: From `overrides` input
  - return_value: The return value of the task function
  - cfg: The configuration object sent to the task function
  - hydra_cfg: The hydra configuration object
  - working_dir: The experiment working directory
  - task_name: The task name of the Hydra job

Next, a Hydra Multirun to sweep over parameters using the Hydra CLI::

    $  python my_app.py bar=mom,dad --multirun
    [2021-05-08 21:07:10,209][HYDRA] Launching 2 jobs locally
    [2021-05-08 21:07:10,209][HYDRA]        #0 : bar=mom
    hello mom
    [2021-05-08 21:07:10,279][HYDRA]        #1 : bar=dad
    hello dad

becomes::

    >>> from my_app_zen import MyExperiment, task_function
    >>> from hydra_zen.experimental import hydra_multirun
    >>> job = hydra_multirun(MyExperiment, task_function, overrides=["bar=mom,dad"])
    [2021-05-08 21:04:35,898][HYDRA] Launching 2 jobs locally
    [2021-05-08 21:04:35,899][HYDRA]        #0 : bar=mom
    hello mom
    [2021-05-08 21:04:35,980][HYDRA]        #1 : bar=dad
    hello dad

An important note, since these functions end up executing Hydra we get all the benefits of Hydra's job configuration and logging.
See Configuring Hydra [2]_ for more details on customizing Hydra.

Examples
========

Return a Simple Dictionary
***********************************

Here we demonstrate some simple examples of running Hydra experiments with hydra-zen.
First lets illustrate the behavior of using ``builds``, ``instantiate``, and ``hydra_run``:

.. code:: python

    >>> from hydra_zen import instantiate, builds
    >>> from hydra_zen.experimental import hydra_run
    >>> job = hydra_run(builds(dict, a=1, b=1), task_function=instantiate)
    >>> job.return_value
    {'a': 1, 'b': 1}

As expected, ``hydra_run`` simply instantiates and creates a dictionary object with the desired key and value pairs.
Next, launch a Hydra multi-run [3]_ job to sweep over configuration parameters:

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


Using Partial Functions
********************************

Now lets demonstrate the use of building partial configurations with hydra-zen.
Here we build a configuration to square an input using the ``pow`` function:

.. code:: python

    >>> from hydra_zen.experimental import hydra_run
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=10)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> job = hydra_run(cfg, task_function)
    >>> job.return_value
    100

Notice the task function must instantiate the partial function and then evaluate on the desired parameter ``x``.
Here is the same code using a multirun experiment:

.. code:: python

    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> jobs = hydra_multirun(cfg, task_function, overrides=["x=range(-2,3)"])
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]


Random Search Optimization
************************************

This example shows how to build a Hydra Sweeper [4]_ for doing random search optimization with hydra-zen.
First lets build the Hydra Sweeper function:

.. code:: python

    import random
    from typing import List, Optional

    from hydra import TaskFunction
    from hydra.core.config_loader import ConfigLoader
    from hydra.core.override_parser.overrides_parser import OverridesParser
    from hydra.plugins.sweeper import Sweeper
    from hydra_zen import instantiate
    from omegaconf import DictConfig


    class RandomSearchSweeper(Sweeper):
        def __init__(self, optim: DictConfig):
            self.opt_config = optim
            self.config: Optional[DictConfig] = None
            self.launcher: Optional[Launcher] = None
            self.job_idx: Optional[int] = None

        def setup(
            self,
            config: DictConfig,
            config_loader: ConfigLoader,
            task_function: TaskFunction,
        ) -> None:
            self.job_idx = 0
            self.config = config
            self.config_loader = config_loader

            self.launcher = instantiate(config.hydra.launcher)
            # assert isinstance(launcher, Launcher)
            self.launcher.setup(
                config=config,
                config_loader=config_loader,
                task_function=task_function,
            )

        def sweep(self, arguments: List[str]) -> None:
            parser = OverridesParser.create()
            parsed = parser.parse_overrides(arguments)

            param_bounds = {}
            for override in parsed:
                key = override.get_key_element()
                val = override.value()
                if override.is_interval_sweep():
                    param_bounds[key] = [val.start, val.end]

            direction = -1 if self.opt_config.maximize else 1
            name = "maximization" if self.opt_config.maximize else "minimization"

            all_results = []
            best_score = None
            best_solution = None
            iterations = 0
            while iterations < self.opt_config.max_iter:
                new_solution = {k: random.uniform(*v) for k, v in param_bounds.items()}

                overrides = [tuple(f"{k}={float(v)}" for k, v in new_solution.items())]

                returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
                score = returns[0].return_value

                if best_score is None:
                    best_score = score
                    best_solution = new_solution
                elif score > best_score and self.opt_config.maximize:
                    best_score = score
                    best_solution = new_solution
                elif score < best_score and not self.opt_config.maximize:
                    best_score = score
                    best_solution = new_solution

                self.job_idx += len(returns)
                iterations += 1
                all_results.append(new_solution)

            results_to_serialize = {
                "best_evaluated_params": best_solution,
                "best_evaluated_result": best_score,
            }
            return results_to_serialize, all_results

If we are to configure our Hydra application to use this sweeper we must store the a configuration in Hydra's Config Store API [5]_

.. code:: python

    from hydra.core.config_store import ConfigStore
    from hydra_zen import builds

    RandomSearchSweeperConf = builds(
        RandomSearchSweeper, optim=dict(x=0, y=12, maximize=False, max_iter=10)
    )

    cs = ConfigStore.instance()
    cs.store(group="hydra/sweeper", name="test_sweeper", node=RandomSearchSweeperConf)

Next lets build the function to minimize:

.. code:: python

    def square(x, y, z=12):
        return (x - 2) ** 2 + (y + 3) ** 2 + abs(z)

    def task_function(cfg):
        return square(cfg.x, cfg.y)

Now we can use ``hydra_multirun`` to minimize this function via random search:

.. code:: python

    from hydra_zen.experimental import hydra_multirun

    job = hydra_multirun(
        dict(x=1, y=12),
        task_function,
        overrides=[
            "hydra/sweeper=test_sweeper",
            "x=interval(-12, 12)",
            "y=interval(-12, 12)",
        ],
    )

The return value is the solution along with all the intermediate results:

.. code:: python

    >>> job.return_value
    ({'best_evaluated_params': {'x': 1.4667823674518203, 'y': -1.193575968925213},
    'best_evaluated_result': 15.547488823704768},
    [{'x': 6.677158827229356, 'y': 9.118453646915974},
    {'x': 5.6361529551972716, 'y': -3.792476421542583},
    {'x': -7.748308725314637, 'y': -5.912971582716322},
    {'x': 8.147470889587407, 'y': 2.0498442440287974},
    {'x': 3.935601640352349, 'y': -4.953283602632575},
    {'x': 10.990198420450358, 'y': -7.984788508753781},
    {'x': -0.022198517203456447, 'y': 5.523420437678112},
    {'x': 1.4667823674518203, 'y': -1.193575968925213},
    {'x': -9.637746564557986, 'y': 11.494829524133394},
    {'x': 9.862498442551928, 'y': -9.1633611350283}])



References
==========
.. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro
.. [2] https://hydra.cc/docs/configure_hydra/intro
.. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
.. [4] https://hydra.cc/docs/next/advanced/plugins
.. [5] https://hydra.cc/docs/next/tutorials/structured_config/config_store