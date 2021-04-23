# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, run_job
from hydra.experimental import compose, initialize
from hydra.plugins.sweeper import Sweeper
from omegaconf import DictConfig, OmegaConf

from .._hydra_overloads import instantiate
from ..typing import DataClass


def _store_config(
    cfg: Union[DataClass, DictConfig, Mapping], config_name: str = "hydra_launch"
) -> str:
    """Generates a Structured Config and registers it in the ConfigStore.

    Parameters
    ----------
    cfg: Union[DataClass, DictConfig, Mapping]
        A configuration as a dataclass, configuration object, or a dictionary.

    config_name: str (default: hydra_launch)
        A default configuration name if available, otherwise a new object is

    Returns
    -------
    config_name: str
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


def _load_config(
    config_name: Optional[str] = None, overrides: Optional[List[str]] = None
) -> DictConfig:
    """Generates the configuration object including Hydra configurations.

    Parameters
    ----------
    config_name: Optional[str] (default: None)
        The configuration name used to store the default configuration.

    overrides: Optional[List[str]] (default: None)
        If provided, overrides default configurations, see [2]_ and [3]_.

    Returns
    -------
    config: DictConfig

    Notes
    -----
    This function uses Hydra's Compose API [1]_

    References
    ----------
    .. [1] https://hydra.cc/docs/experimental/compose_api
    .. [2] https://hydra.cc/docs/configure_hydra/intro
    .. [3] https://hydra.cc/docs/advanced/override_grammar/basic
    """

    with initialize():
        task_cfg = compose(
            config_name,
            overrides=[] if overrides is None else overrides,
            return_hydra_config=True,
        )

    return task_cfg


def hydra_run(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    overrides: Optional[List[str]] = None,
    config_name: str = "hydra_run",
) -> JobReturn:
    """Launch a Hydra job defined by ``task_function`` using the configuration
    provided in ``config``.

    Similar to how Hydra CLI works, ``overrides`` are a string list of configuration
    values to use for a given experiment run.  For example, the Hydra CLI provided by

    $ python -m job.task_function job/group=group_name job.group.param=1

    would be

    >>> job = hydra_run(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

    This functions executes Hydra and therefore creates its own working directory.  See Configuring Hydra [2]_ for more
    details on customizing Hydra.

    Parameters
    ----------
    config: Union[DataClass, DictConfig]
        The experiment configuration.

    task_function: Callable[[DictConfig], Any]
        The function Hydra will execute with the given configuration.

    overrides: Optional[List[str]] (default: None)
        If provided, overrides default configurations, see [1]_ and [2]_.

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    job_name: str (default: "hydra_run")

    Returns
    -------
    result: JobReturn
        The object storing the results of the Hydra experiment.
            - overrides: From ``overrides`` and ``multirun_overrides``
            - return_value: The return value of the task function
            - cfg: The configuration object sent to the task function
            - hydra_cfg: The hydra configuration object
            - working_dir: The experiment working directory
            - task_name: The task name of the Hydra job

    References
    ----------
    .. [1] https://hydra.cc/docs/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/configure_hydra/intro

    Examples
    --------

    Simple Hydra ``run``:

    >>> from hydra_zen import instantiate, builds
    >>> from hydra_zen.experimental import hydra_run
    >>> job = hydra_run(builds(dict, a=1, b=1), task_function=lambda x: instantiate(x))
    >>> job.return_value
    {'a': 1, 'b': 1}

    Using a more complex ``task_function``

    >>> from hydra_zen.experimental import hydra_run
    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

    Launch a job to evaluate the function using the given configuration:

    >>> job = hydra_run(cfg, task_function)
    >>> job.return_value
    1

    An example using ``PyTorch``

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, hydra_partial=True)
    >>> ModelConfig = builds(Linear, in_features=1, out_features=1)
    >>> cfg = dict(optim=AdamConfig(), model=ModelConfig())
    >>> def task_function(cfg):
    ...     model = instantiate(cfg.model)
    ...     optim = instantiate(cfg.optim)(model.parameters())
    ...     loss = model(torch.ones(1)).mean()
    ...     optim.zero_grad()
    ...     loss.backward()
    ...     optim.step()
    ...     return loss.item()
    >>> jobs = hydra_run(cfg, task_function, overrides=["optim.lr=0.1"])
    >>> j.return_value
    0.3054758310317993
    """
    config_name = _store_config(config, config_name)
    task_cfg = _load_config(config_name=config_name, overrides=overrides)

    try:
        job = run_job(
            config=task_cfg,
            task_function=task_function,
            job_dir_key="hydra.run.dir",
            job_subdir_key=None,
            configure_logging=False,
        )
    finally:
        GlobalHydra.instance().clear()
    return job


def hydra_multirun(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    config_name: str = "hydra_multirun",
    job_name: str = "hydra_multirun",
) -> List[JobReturn]:
    """Launch a Hydra ``multirun`` ([1]_) job defined by ``task_function`` using the configuration
    provided in ``config``.

    Similar to how Hydra CLI works, ``overrides`` are a string list of configuration
    values to use for a given experiment run.  For example, the Hydra CLI provided by::

       $ python -m job.task_function job/group=group_name job.group.param=1 --multirun

    would be::

       >>> job = hydra_multirun(config, task_function, overrides=["job/group=group_name", "job.group.param=1"])

    To sweep over parameters the Hydra CLI provided by

    $ python -m job.task_function job/group=group_name job.group.param=1,2,3 --multirun

    would be

    >>> job = hydra_multirun(config, task_function, overrides=["job/group=group_name", "job.group.param=1,2,3"])

    This functions executes Hydra and therefore creates its own working directory.  See Configuring Hydra [3]_ for more
    details on customizing Hydra.

    Parameters
    ----------
    config: Union[DataClass, DictConfig]
        The experiment configuration.

    task_function: Callable[[DictConfig], Any]
        The function Hydra will execute with the given configuration.

    overrides: Optional[List[str]] (default: None)
        If provided, overrides default configurations, see [2]_ and [3]_.

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    job_name: str (default: "hydra_multirun")

    Returns
    -------
    result: List[List[JobReturn]]
        The object storing the results of each Hydra experiment.
            - overrides: From ``overrides`` and ``multirun_overrides``
            - return_value: The return value of the task function
            - cfg: The configuration object sent to the task function
            - hydra_cfg: The hydra configuration object
            - working_dir: The experiment working directory
            - task_name: The task name of the Hydra job

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
    .. [2] https://hydra.cc/docs/advanced/override_grammar/basic
    .. [3] https://hydra.cc/docs/configure_hydra/intro

    Examples
    --------

    Simple Hydra `multirun``:

    >>> job = hydra_multirun(builds(dict, a=1, b=1), task_function=lambda x: instantiate(x), overrides=["a=1,2"])
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]

    Using a more complex ``task_function``

    >>> from hydra_zen import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

    Launch a ``multirun`` over a list of different ``x`` values using Hydra's override syntax ``range``:

    >>> jobs = hydra_multirun(cfg, task_function, overrides=["x=range(-2,3)"])
    >>> [j.return_value for j in jobs[0]]
    [4, 1, 0, 1, 4]

    An example using ``PyTorch``

    >>> from torch.optim import Adam
    >>> from torch.nn import Linear
    >>> AdamConfig = builds(Adam, lr=0.001, hydra_partial=True)
    >>> ModelConfig = builds(Linear, in_features=1, out_features=1)
    >>> cfg = dict(optim=AdamConfig(), model=ModelConfig())
    >>> def task_function(cfg):
    ...     model = instantiate(cfg.model)
    ...     optim = instantiate(cfg.optim)(model.parameters())
    ...     loss = model(torch.ones(1)).mean()
    ...     optim.zero_grad()
    ...     loss.backward()
    ...     optim.step()
    ...     return loss.item()

    Evaluate the function for different learning rates

    >>> jobs = hydra_multirun(cfg, task_function, overrides=["optim.lr=0.1,1.0"])
    >>> [j.return_value for j in jobs[0]]
    [0.3054758310317993, 0.28910207748413086]
    """

    # Separate Hydra overrides from experiment overrides
    hydra_overrides = []
    _overrides = []
    for o in overrides:
        if o.startswith("hydra"):
            hydra_overrides.append(o)
        else:
            _overrides.append(o)

    # Only the hydra overrides are needed to extract the Hydra configuration for
    # the launcher and sweepers.
    # The sweeper handles the overides for each experiment
    config_name = _store_config(config, config_name)
    task_cfg = _load_config(config_name=config_name, overrides=hydra_overrides)

    if config_dir is not None:
        config_dir = str(Path(config_dir).absolute())
    search_path = create_config_search_path(config_dir)

    hydra = Hydra.create_main_hydra2(task_name=job_name, config_search_path=search_path)
    try:
        # Instantiate sweeper without using Hydra's Plugin discovery
        sweeper = instantiate(task_cfg.hydra.sweeper)
        assert isinstance(sweeper, Sweeper)
        sweeper.setup(
            config=task_cfg,
            config_loader=hydra.config_loader,
            task_function=task_function,
        )

        # Obtain any overrides set by user (need all + multirun params)
        _overrides += task_cfg.hydra.overrides.task

        # just ensures repeats are removed
        _overrides = list(set(_overrides))
        job = sweeper.sweep(arguments=_overrides)
    finally:
        GlobalHydra.instance().clear()
    return job
