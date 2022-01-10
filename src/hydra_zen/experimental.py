# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import warnings
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Union

from omegaconf.dictconfig import DictConfig

from hydra_zen.errors import HydraZenDeprecationWarning
from hydra_zen.typing._implementations import DataClass

from ._launch import launch


def hydra_run(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    config_name: str = "hydra_run",
    job_name: str = "hydra_run",
    with_log_configuration: bool = True,
):
    """(Deprecated) Launch a Hydra job defined by `task_function` using the configuration
    provided in `config`.

    .. deprecated:: 0.3.0
          `hydra_run` will be removed in hydra-zen 1.0.0; it is replaced by
          :func:`hydra_zen.launch`.

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

    Returns
    -------
    result : JobReturn

    References
    ----------
    .. [1] https://hydra.cc/docs/next/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/next/configure_hydra/intro
    .. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
    """
    warnings.warn(
        HydraZenDeprecationWarning(
            "hydra_zen.experimental.hydra_run is deprecated "
            "as of 2021-10-27. Change `hydra_run(cfg, task_fn, overrides, ...)` to `launch(cfg, task_fn, overrides, ...)`."
            "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-27 — whichever "
            "comes first.\n\nNote: This deprecation does not impact yaml configs "
            "produced by `builds`."
        ),
        stacklevel=2,
    )
    return launch(
        config,
        task_function=task_function,
        overrides=overrides,
        config_dir=config_dir,
        config_name=config_name,
        job_name=job_name,
        with_log_configuration=with_log_configuration,
    )


def hydra_multirun(
    config: Union[DataClass, DictConfig, Mapping],
    task_function: Callable[[DictConfig], Any],
    overrides: Optional[List[str]] = None,
    config_dir: Optional[Union[str, Path]] = None,
    config_name: str = "hydra_run",
    job_name: str = "hydra_run",
    with_log_configuration: bool = True,
):
    """(Deprecated) Launch multiple Hydra jobs defined by `task_function` using the configuration
    provided in `config`.

    .. deprecated:: 0.3.0
          `hydra_multirun` will be removed in hydra-zen 1.0.0; it is replaced by
          :func:`hydra_zen.launch`.

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

    Returns
    -------
    result : Any

    References
    ----------
    .. [1] https://hydra.cc/docs/next/advanced/override_grammar/basic
    .. [2] https://hydra.cc/docs/next/configure_hydra/intro
    .. [3] https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run
    """
    warnings.warn(
        HydraZenDeprecationWarning(
            "hydra_zen.experimental.hydra_multirun is deprecated "
            "as of 2021-10-27. Change `hydra_multirun(cfg, task_fn, overrides, ...)` to `launch(cfg, task_fn, overrides, multirun=True, ...)`."
            "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-27 — whichever "
            "comes first.\n\nNote: This deprecation does not impact yaml configs "
            "produced by `builds`."
        ),
        stacklevel=2,
    )
    return launch(
        config,
        task_function=task_function,
        overrides=overrides,
        config_dir=config_dir,
        config_name=config_name,
        job_name=job_name,
        with_log_configuration=with_log_configuration,
        multirun=True,
    )
