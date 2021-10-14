# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

from hydra._internal.callbacks import Callbacks
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.utils import JobReturn
from hydra.plugins.sweeper import Sweeper
from hydra.types import HydraContext, RunMode
from omegaconf import DictConfig, OmegaConf

from hydra_zen._hydra_overloads import instantiate
from hydra_zen.typing._implementations import DataClass


def _store_config(
    cfg: Union[DataClass, DictConfig, Mapping], config_name: str = "hydra_launch"
) -> str:
    """Stores configuration object in Hydra's ConfigStore.

    Parameters
    ----------
    cfg : Union[DataClass, DictConfig, Mapping]
        A configuration as a dataclass, configuration object, or a dictionary.

    config_name : str (default: hydra_launch)
        The configuration name used to store the configuration.

    Returns
    -------
    config_name : str
        The configuration name used to store the default configuration.

    Notes
    -----
    The input configuration is registered in the Hydra ConfigStore [1]_ using a
    user-provided config name.

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/structured_config/config_store
    """
    cs = ConfigStore().instance()
    cs.store(name=config_name, node=cfg)
    return config_name


def launch(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    config_name: str = "_zen_launch",
    job_name: str = "_zen_launch",
    with_log_configuration: bool = True,
    multirun: bool = False,
) -> Union[JobReturn, Any]:
    """Launch a Hydra job defined by `task_function` using the configuration
    provided in `config`.

    Similar to how Hydra CLI works, `overrides` are a string list of configuration
    values to use for a given experiment run.  For example, the Hydra CLI provided by::

       $ python my_task.py job/group=group_name job.group.param=1

    would be::

       >>> job = launch(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

    This functions executes Hydra and therefore creates its own working directory.  See Configuring Hydra [2]_ for more
    details on customizing Hydra.

    Similarily, to launch a `multirun` job and sweep over parameters the Hydra CLI provided by::

       $ python -m job.task_function job/group=group_name job.group.param=1,2,3 --multirun

    would become::

       >>> job = launch(config, task_function, overrides=["job/group=group_name", "job.group.param=1,2,3"], multirun=True)


    Parameters
    ----------
    config : Union[DataClass, DictConfig, Mapping]
        A configuration as a dataclass, configuration object, or a dictionary.

    task_function : Callable[[DictConfig], Any]
        The function Hydra will execute with the given configuration.

    overrides : Optional[List[str]] (default: None)
        If provided, overrides default configurations, see [1]_ and [2]_.

    config_dir : Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    config_name : str (default: "hydra_run")
        Name of the stored configuration in Hydra's ConfigStore API.

    job_name : str (default: "hydra_run")

    with_log_configuration : bool (default: True)
        Flag to configure logging subsystem from the loaded config

    multirun : bool (default: False)
        Launch a Hydra multi-run ([3]_)

    Returns
    -------
    result : JobReturn | Any
        If ``multirun = False``:
            A ``JobReturn`` object storing the results of the Hydra experiment.
                - overrides: From `overrides` and `multirun_overrides`
                - return_value: The return value of the task function
                - cfg: The configuration object sent to the task function
                - hydra_cfg: The hydra configuration object
                - working_dir: The experiment working directory
                - task_name: The task name of the Hydra job
        Else:
            Return values of all launched jobs (depends on the Sweeper implementation).

    References
    ----------
    .. [1] https://hydra.cc/docs/next/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/next/configure_hydra/intro
    .. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run

    Examples
    --------
    Simple Hydra run:

    >>> from hydra_zen import instantiate, builds, launch
    >>> job = launch(builds(dict, a=1, b=1), task_function=instantiate)
    >>> job.return_value
    {'a': 1, 'b': 1}

    Using a more complex task function:

    >>> cfg = dict(f=builds(pow, exp=2, zen_partial=True), x=10)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

    Launch a job to evaluate the function using the given configuration:

    >>> job = launch(cfg, task_function)
    >>> job.return_value
    100

    Next, a Hydra multi-run can be launched by setting `multirun=True`:

    >>> job = launch(
    ...     builds(dict, a=1, b=1),
    ...     task_function=instantiate,
    ...     overrides=["a=1,2"],
    ...     multirun=True
    ... )
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]

    Launch a multi-run over a list of different `x` values using Hydra's override syntax `range`:

    >>> jobs = launch(cfg, task_function, overrides=["x=range(-2,3)"], multirun=True)
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]

    An example using PyTorch

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, zen_partial=True)
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

    >>> jobs = launch(cfg, task_function, overrides=["optim.lr=0.1,1.0"], multirun=True)
    >>> [j.return_value for j in jobs[0]]
    [0.3054758310317993, 0.28910207748413086]
    """
    config_name = _store_config(config, config_name)

    if config_dir is not None:
        config_dir = str(Path(config_dir).absolute())
    search_path = create_config_search_path(config_dir)

    hydra = Hydra.create_main_hydra2(task_name=job_name, config_search_path=search_path)
    try:
        if not multirun:
            # Here we can use Hydra's `run` method
            job = hydra.run(
                config_name=config_name,
                task_function=task_function,
                overrides=overrides if overrides is not None else [],
                with_log_configuration=with_log_configuration,
            )

        else:
            # Instead of running Hydra's `multirun` method we instantiate
            # and run the sweeper method.  This allows us to run local
            # sweepers and launchers without installing them in `hydra_plugins`
            # package directory.
            cfg = hydra.compose_config(
                config_name=config_name,
                overrides=overrides if overrides is not None else [],
                with_log_configuration=with_log_configuration,
                run_mode=RunMode.MULTIRUN,
            )

            callbacks = Callbacks(cfg)
            callbacks.on_multirun_start(config=cfg, config_name=config_name)

            # Instantiate sweeper without using Hydra's Plugin discovery (Zen!)
            sweeper = instantiate(cfg.hydra.sweeper)
            assert isinstance(sweeper, Sweeper)
            sweeper.setup(
                config=cfg,
                hydra_context=HydraContext(
                    config_loader=hydra.config_loader, callbacks=callbacks
                ),
                task_function=task_function,
            )

            task_overrides = OmegaConf.to_container(
                cfg.hydra.overrides.task, resolve=False
            )
            assert isinstance(task_overrides, list)

            job = sweeper.sweep(arguments=task_overrides)
            callbacks.on_multirun_end(config=cfg, config_name=config_name)

    finally:
        GlobalHydra.instance().clear()
    return job
