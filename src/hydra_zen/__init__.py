# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import TYPE_CHECKING

from ._hydra_overloads import (
    MISSING,
    instantiate,
    load_from_yaml,
    save_as_yaml,
    to_yaml,
)
from ._launch import hydra_list, launch, multirun
from .structured_configs import (
    ZenField,
    builds,
    hydrated_dataclass,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
)
from .structured_configs._implementations import (
    BuildsFn,
    DefaultBuilds,
    get_target,
    kwargs_of,
)
from .structured_configs._type_guards import is_partial_builds, uses_zen_processing
from .wrapper import ZenStore, store, zen

__all__ = [
    "builds",
    "BuildsFn",
    "DefaultBuilds",
    "hydrated_dataclass",
    "just",
    "kwargs_of",
    "mutable_value",
    "get_target",
    "MISSING",
    "instantiate",
    "load_from_yaml",
    "save_as_yaml",
    "to_yaml",
    "make_config",
    "ZenField",
    "make_custom_builds_fn",
    "launch",
    "is_partial_builds",
    "uses_zen_processing",
    "zen",
    "hydra_list",
    "multirun",
    "store",
    "ZenStore",
]

if not TYPE_CHECKING:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "unknown version"
else:  # pragma: no cover
    __version__: str
