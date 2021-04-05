# Copyright (c) 2021 Massachusetts Institute of Technology
import copy
import string
from dataclasses import is_dataclass, make_dataclass
from pathlib import Path
from random import choice
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from hydra._internal.config_loader_impl import ConfigLoaderImpl
from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_config_search_path
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore
from hydra.core.global_hydra import GlobalHydra
from hydra.core.plugins import Plugins
from hydra.core.utils import run_job
from hydra.types import RunMode
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.omegaconf import open_dict

from hydra_utils.typing import DataClass


def get_hydra_cfg(
    *, overrides: Optional[List[str]] = None, config_dir: Optional[str] = None
) -> HydraConf:
    """Generates the Hydra Configuration.

    Parameters
    ----------
    overrides: Optional[List[str]]
        Overrides as a list of "dot" configs for the hydra config (e.g., ["hydra.run.dir='here'"]).

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    Returns
    -------
    hydra_cfg: DictConfig
        A HydraConf configuration object

    """
    config_search_path = create_config_search_path(config_dir)
    config_loader = ConfigLoaderImpl(config_search_path=config_search_path)

    # TODO: Does RunMode matter for this??
    cfg = config_loader.load_configuration(
        config_name=None,
        overrides=[] if overrides is None else overrides,
        run_mode=RunMode.RUN,
        from_shell=False,  # tries to handle bash if true
    )

    # return Hydra.get_sanitized_hydra_cfg(cfg)
    return HydraConf(**cfg.hydra)


def _gen_config(
    cfg: DictConfig, config_name: Optional[str] = None
) -> Tuple[Union[DictConfig, ListConfig], str]:
    """Generates a Structured Config and registers it in the ConfigStore.

    Notes
    -----
    Using Hydra Sweeper/Launcher Plugins requires a config (yaml or structured) to load the defaults.

    This class generates a config with defaults based on the values in the input config.

    The generated config is registered in the Hydra ConfigStore using a randomly generated or user provided config name.

    Parameters
    ----------
    cfg: DictConfig

    config_name: Optional[str]

    Returns
    -------
    config: Dictconfig
    config_name: str

    """
    if config_name is None:
        # TODO: Too much??
        letters = string.ascii_lowercase
        config_name = "".join(choice(letters) for i in range(10))

    # @dataclass
    # class MultirunConfig:
    #     raiden: RaidenConf = cfg.raiden
    #     seed: int = cfg.seed
    #     testing: bool = cfg.testing

    MultirunConfig = make_dataclass(
        "MultirunConfig",
        [(k, OmegaConf.get_type(v), v) for k, v in cfg.items() if k != "hydra"],
    )

    cs = ConfigStore.instance()
    cs.store(name=config_name, node=MultirunConfig)
    # the parameters are already stored in the config
    out = OmegaConf.create(MultirunConfig())
    return out, config_name


def hydra_launch(
    config: Union[DataClass, DictConfig],
    task_function: Callable[[Any], Any],
    multirun_overrides: Optional[Sequence[str]] = None,
    hydra_overrides: Optional[Sequence[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    job_name: str = "hydra_launch",
) -> Any:
    """Launch Hydra job.

    Parameters
    ----------
    config: Union[DataClass, DictConfig]
        The experiment configuration.

    task_function: Callable[[Any], Any]
        The function Hydra will execute with the given configuration.

    multirun_overrides: Optional[Sequence[str]] (default: None)
        If provided, Hydra will run in "multirun" mode using the provided overrides.

    hydra_overrides: Optional[Sequence[str]] (default: None)
        If provided, updates

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    job_name: str (default: "hydra_launch")

    Returns
    -------
    result: Any
        The return value of the task_function

    Examples
    --------
    >>> from hydra_utils import instantiate, builds
    >>> from hydra_utils.experimental import hydra_launch
    >>> def f(a: int = 1, b: int = 2):
    ...    return dict(a=a, b=b)

    >>> job = hydra_launch(builds(f, a=1, b=1), task_function=lambda x: instantiate(x))
    >>> job.return_value
    {'a': 1, 'b': 1}

    >>> def f(a: int = 1, b: int = 2):
    ...    return dict(a=a, b=b)
    >>> hydra, job = hydra_launch(builds(f, a=1, b=1), task_function=lambda x: instantiate(x), multirun_overrides=["a=1,2"])
    >>> [j.return_value for j in job[0]]
    [{'a': 1, 'b': 1}, {'a': 2, 'b': 1}]
    """
    if is_dataclass(config):
        task_cfg = OmegaConf.create(config)
    else:
        task_cfg = copy.deepcopy(config)

    if "hydra" in task_cfg and hydra_overrides and len(hydra_overrides) > 0:
        raise ValueError(
            "hydra_overrides set when hydra config is already provided in the input config"
        )

    if config_dir is not None:
        config_dir = str(Path(config_dir).absolute())
    search_path = create_config_search_path(config_dir)

    hydra = Hydra.create_main_hydra2(task_name=job_name, config_search_path=search_path)
    try:
        if "hydra" not in task_cfg:
            task_cfg, config_name = _gen_config(task_cfg)
            hydra_cfg = get_hydra_cfg(overrides=hydra_overrides, config_dir=config_dir)
            with open_dict(task_cfg):
                task_cfg = OmegaConf.merge(
                    task_cfg,
                    dict(hydra=OmegaConf.create(hydra_cfg)),
                )
            task_cfg.hydra.job.config_name = config_name

        multirun = multirun_overrides is not None and len(multirun_overrides) > 0
        if multirun:
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
