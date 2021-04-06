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
from hydra.core.utils import run_job, JobReturn
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
    config, config_name: Tuple[Union[DictConfig, ListConfig], str]
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

    # TODO: Need better type handling?
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
    task_function: Callable[[DictConfig], Any],
    multirun_overrides: Optional[List[str]] = None,
    hydra_overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    job_name: str = "hydra_launch",
) -> JobReturn:
    """Launch Hydra job.

    Parameters
    ----------
    config: Union[DataClass, DictConfig]
        The experiment configuration.

    task_function: Callable[[DictConfig], Any]
        The function Hydra will execute with the given configuration.

    multirun_overrides: Optional[List[str]] (default: None)
        If provided, Hydra will run in "multirun" mode using the provided overrides.  See [here](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run).

    hydra_overrides: Optional[List[str]] (default: None)
        If provided, overrides default hydra configurations. See [here](https://hydra.cc/docs/advanced/override_grammar/basic).

    config_dir: Optional[Union[str, Path]] (default: None)
        Add configuration directories if needed.

    job_name: str (default: "hydra_launch")

    Returns
    -------
    result: JobReturn
        The return value of the task_function

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
    ...     return loss

    Evalulate the function for different learning rates

    >>> jobs = hydra_launch(cfg, task_function, multirun_overrides=["optim.lr=0.1,1.0"])
    >>> [j.return_value for j in jobs[0]]
    [tensor(0.1803, grad_fn=<MeanBackward0>),
    tensor(-0.2261, grad_fn=<MeanBackward0>)]

    To configuring Hydra options via ``hydra_overrides`` see [here](https://hydra.cc/docs/configure_hydra/intro)
    """
    if is_dataclass(config):
        task_cfg = OmegaConf.create(config)
    else:
        task_cfg = copy.deepcopy(config)

    if "hydra" in task_cfg and hydra_overrides:
        raise ValueError(
            "`hydra_overrides` cannot be specified when `config` is already derived from a HydraConfig"
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
