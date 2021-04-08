# Copyright (c) 2021 Massachusetts Institute of Technology
import copy
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.core.utils import JobReturn, run_job
from hydra.experimental import compose, initialize
from omegaconf import DictConfig, OmegaConf

from hydra_utils.typing import DataClass


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
    The input configuration is registered in the Hydra ConfigStore [1]_ using a randomly
    generated or user provided config name.

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


def hydra_launch(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    multirun_overrides: Optional[List[str]] = None,
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    config_name: str = "hydra_launch",
    job_name: str = "hydra_launch",
) -> JobReturn:
    """Launch Hydra job.

    Parameters
    ----------
    config: Union[DataClass, DictConfig]
        The experiment configuration.

    task_function: Callable[[DictConfig], Any]
        The function Hydra will execute with the given configuration.

    overrides: Optional[List[str]] (default: None)
        If provided, overrides default configurations, see [2]_ and [3]_.

    multirun_overrides: Optional[List[str]] (default: None)
        If provided, Hydra will run in "multirun" mode using the provided overrides [1]_.

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    job_name: str (default: "hydra_launch")

    Returns
    -------
    result: JobReturn
        The return value of the task_function

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
    .. [2] https://hydra.cc/docs/advanced/override_grammar/basic
    .. [3] https://hydra.cc/docs/configure_hydra/intro

    Examples
    --------

    A simple usage of ``hydra_launch`` to understand the difference between a Hydra ``run`` and ``multirun`` job.

    Simple Hydra ``run``:

    >>> from hydra_utils import instantiate, builds
    >>> from hydra_utils.experimental import hydra_launch
    >>> job = hydra_launch(builds(dict, a=1, b=1), task_function=lambda x: instantiate(x))
    >>> job.return_value
    {'a': 1, 'b': 1}

    Using Hydra ``multirun``:

    >>> job = hydra_launch(builds(dict, a=1, b=1), task_function=lambda x: instantiate(x), multirun_overrides=["a=1,2"])
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]

    Using a more complex ``task_function``

    >>> from hydra_utils.experimental import hydra_launch
    >>> from hydra_utils import builds, instantiate
    >>> cfg = dict(f=builds(pow, exp=2, hydra_partial=True), x=1)
    >>> def task_function(cfg):
    ...    return instantiate(cfg.f)(cfg.x)

    Launch a job to evaluate the function using the given configuration:

    >>> job = hydra_launch(cfg, task_function)
    >>> job.return_value
    1

    Launch a ``multirun`` over a list of different ``x`` values using Hydra's override syntax ``range``:

    >>> jobs = hydra_launch(cfg, task_function, multirun_overrides=["x=range(-2,3)"])
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

    Evalulate the function for different learning rates

    >>> jobs = hydra_launch(cfg, task_function, multirun_overrides=["optim.lr=0.1,1.0"])
    >>> [j.return_value for j in jobs[0]]
    [0.3054758310317993, 0.28910207748413086]
    """
    if not OmegaConf.is_config(config) or not hasattr(config, "hydra"):
        config_name = _store_config(config, config_name)
        task_cfg = _load_config(config_name=config_name, overrides=overrides)
    else:
        overrides = [] if overrides is None else overrides
        if len(overrides) > 0:
            raise ValueError(
                "Non-empty overrides provided with full config object already provided, did you mean `multirun_overrides`?"
            )
        task_cfg = copy.deepcopy(config)

    if config_dir is not None:
        config_dir = str(Path(config_dir).absolute())
    search_path = create_config_search_path(config_dir)

    hydra = Hydra.create_main_hydra2(task_name=job_name, config_search_path=search_path)
    try:
        multirun_overrides = [] if multirun_overrides is None else multirun_overrides
        if len(multirun_overrides) > 0:
            sweeper = Plugins.instance().instantiate_sweeper(
                config=task_cfg,
                config_loader=hydra.config_loader,
                task_function=task_function,
            )

            # Obtain any overrides set by user (need all + multirun params)
            multirun_overrides += task_cfg.hydra.overrides.task

            # just ensures repeats are removed
            multirun_overrides = list(set(multirun_overrides))
            job = sweeper.sweep(arguments=multirun_overrides)
        else:
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
