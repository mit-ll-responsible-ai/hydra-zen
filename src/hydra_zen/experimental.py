# Copyright (c) 2021 Massachusetts Institute of Technology
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
    warnings.warn(
        HydraZenDeprecationWarning(
            "hydra_zen.experimental.hydra_run is deprecated "
            "as of 2021-10-13. Change `hydra_run(cfg, task_fn, overrides, ...)` to `launch(cfg, task_fn, overrides, ...)`."
            "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-13 — whichever "
            "comes first.\n\nNote: This deprecation does not impact yaml configs "
            "produced by `builds`."
        ),
        stacklevel=2,
    )
    return launch(
        config,
        task_function,
        overrides,
        config_dir,
        config_name,
        job_name,
        with_log_configuration,
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
    warnings.warn(
        HydraZenDeprecationWarning(
            "hydra_zen.experimental.hydra_multirun is deprecated "
            "as of 2021-10-13. Change `hydra_multirun(cfg, task_fn, overrides, ...)` to `launch(cfg, task_fn, overrides, multirun=True, ...)`."
            "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-13 — whichever "
            "comes first.\n\nNote: This deprecation does not impact yaml configs "
            "produced by `builds`."
        ),
        stacklevel=2,
    )
    return launch(
        config,
        task_function,
        overrides,
        config_dir,
        config_name,
        job_name,
        with_log_configuration,
        multirun=True,
    )
