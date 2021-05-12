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

hydra-zen includes experimental code to run Hydra jobs purely in Python for both single and multirun experiments.
These functions, ``hydra_run`` and ``hydra_multirun``, are python functions with the following signature:

  - ``config``: A python object of type ``Union[DataClass, DictConfig, Mapping]``
  - ``task_function``: A python function of type ``Callable[[DictConfig], Any]``
  - ``overrides``: A string list of overrides that mimic Hydra Command Line Interface (CLI) or Hydra Compose API

These functions initialize and run Hydra and therefore provide all the benefits of Hydra's job and logging configurations.
See Configuring Hydra [2]_ for more details on customizing Hydra.

To demonstrate the difference between running Hydra CLI and hydra-zen start by defining a Hydra application:

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

Using the Hydra CLI, the application can be run with following command::

    $  python my_app.py bar=mom
    hello mom

To execute the same command in a Python environment using ``hydra_run``, simply execute the following:

  >>> from my_app import MyExperiment, task_function
  >>> from hydra_zen.experimental import hydra_run
  >>> job = hydra_run(MyExperiment, task_function, overrides=["bar=mom"])
  hello mom

In addition to running the experiments in a Pythonic way, we also have the return object from the experiment.
For the above application the return object is a Hydra ``JobReturn`` object with the following attributes:

  - overrides: From `overrides` input
  - return_value: The return value of the task function
  - cfg: The configuration object sent to the task function
  - hydra_cfg: The hydra configuration object
  - working_dir: The experiment working directory
  - task_name: The task name of the Hydra job

Next, to run a Hydra Multirun [3]_ job and sweep over parameters using the Hydra CLI::

    $  python my_app.py bar=mom,dad --multirun
    [2021-05-08 21:07:10,209][HYDRA] Launching 2 jobs locally
    [2021-05-08 21:07:10,209][HYDRA]        #0 : bar=mom
    hello mom
    [2021-05-08 21:07:10,279][HYDRA]        #1 : bar=dad
    hello dad

The equivalent ``hydra_multirun`` commands are::

    >>> from my_app import MyExperiment, task_function
    >>> from hydra_zen.experimental import hydra_multirun
    >>> job = hydra_multirun(MyExperiment, task_function, overrides=["bar=mom,dad"])
    [2021-05-08 21:04:35,898][HYDRA] Launching 2 jobs locally
    [2021-05-08 21:04:35,899][HYDRA]        #0 : bar=mom
    hello mom
    [2021-05-08 21:04:35,980][HYDRA]        #1 : bar=dad
    hello dad

Configurations
**************

The simplest way to create configuration is to use ``builds`` to generate dataclasses or just use simple dictionaries.
First let's demonstrate using ``builds``.

.. code:: python

    >>> from hydra_zen import instantiate, builds
    >>> from hydra_zen.experimental import hydra_run
    >>> job = hydra_run(builds(dict, a=1, b=1), task_function=instantiate)
    >>> job.return_value
    {'a': 1, 'b': 1}

As expected, ``hydra_run`` simply instantiates and creates a dictionary object with the desired key and value pairs.
Next, launch a Hydra Multirun [3]_ job to sweep over configuration parameters:

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


Now let's demonstrate building a dictionary configuration.
Here we build a configuration to square an input using the ``pow`` function:

.. code:: python

    >>> from omegaconf import DictConfig
    >>> from hydra_zen.experimental import hydra_run
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=10)
    >>> def task_function(cfg: DictConfig):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> job = hydra_run(cfg, task_function)
    >>> job.return_value
    100

First note that the input to the task function is a configuration object, ``DictConfig``.
Also, the task function must instantiate the partial function and then evaluate on the desired parameter ``x``.
Here is the same code using a multirun experiment:

.. code:: python

    >>> from omegaconf import DictConfig
    >>> from hydra_zen.experimental import hydra_multirun
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg: DictConfig):
    ...    return instantiate(cfg.f)(cfg.x)
    >>> jobs = hydra_multirun(cfg, task_function, overrides=["x=range(-2,3)"])
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]


Registering Configs for ``overrides``
*************************************

Consider the experiment demonstrated in the ``README`` to minimizing a function using gradient descent:

.. code:: python

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

    # defines our surface that we will be descending
    def parabaloid(x, y):
        return 0.1 * x ** 2 + 0.2 * y ** 2

Instead of running an experiment by varying the momentum of ``SGD``, what if we wanted to vary the optimizer itself?
To do this without ever leaving Python we must take advantage of Hydra's Config Store API [5]_.

.. code:: python

    from hydra.core.config_store import ConfigStore
    from torch.optim import Optimizer, SGD, Adam
    from hydra_zen import builds

    SGDConf = builds(SGD, lr=0.3, momentum=0.0, hydra_partial=True)
    AdamConf = builds(Adam, lr=0.3, hydra_partial=True)

    cs = ConfigStore.instance()
    cs.store(group="optim", name="sgd", node=SGDConf)
    cs.store(group="optim", name="adam", node=AdamConf)

Now define the experiment configuration.

.. code:: python

    ConfigGradDesc = builds(
        gradient_descent,
        optim=None,  # not defined yet
        landscape_fn=parabaloid,
        starting_xy=(-1.5, 0.5),
        num_steps=20,
    )

First, test running the default experiment.

.. code:: python

    >>> from hydra_zen.experimental import hydra_run
    >>> jobs = hydra_multirun(
    ...     ConfigGradDesc,
    ...     task_function=instantiate,
    ...     overrides=["optim=sgd"],
    ... )


Next run the experiment by varying the optimizer.

.. code:: python

    >>> from hydra_zen.experimental import hydra_multirun
    >>> jobs = hydra_multirun(
    ...     ConfigGradDesc,
    ...     task_function=instantiate,
    ...     overrides=["optim=sgd,adam"],
    ... )


Random Search Optimization
**************************

This example shows how to build a Hydra Sweeper [4]_ for doing random search optimization with hydra-zen.
First let's build the Hydra Sweeper function:

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
            *,
            hydra_context: HydraContext,
            task_function: TaskFunction,
            config: DictConfig,
        ) -> None:
            self.job_idx = 0
            self.hydra_context = hydra_context
            self.config = config

            self.launcher = instantiate(config.hydra.launcher)
            self.launcher.setup(
                config=config,
                task_function=task_function,
                hydra_context=hydra_context,
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

            all_results = []
            best_score = None
            best_solution = None
            iterations = 0
            while iterations < self.opt_config.max_iter:
                new_solution = {k: random.uniform(*v) for k, v in param_bounds.items()}

                overrides = [tuple(f"{k}={float(v)}" for k, v in new_solution.items())]

                returns = self.launcher.launch(overrides, initial_job_idx=self.job_idx)
                score = returns[0].return_value

                if best_score is None or score < best_score:
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

To configure our Hydra application with this sweeper we must use Hydra's Config Store API [5]_ to ensure the configuration is available to our Hydra application.

.. code:: python

    from hydra.core.config_store import ConfigStore
    from hydra_zen import builds

    RandomSearchSweeperConf = builds(
        RandomSearchSweeper, optim=dict(x=0, y=12, maximize=False, max_iter=10)
    )

    cs = ConfigStore.instance()
    cs.store(group="hydra/sweeper", name="test_sweeper", node=RandomSearchSweeperConf)

Next let's build the function to minimize and define the task function:

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