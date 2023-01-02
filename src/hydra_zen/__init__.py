# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import TYPE_CHECKING

from ._hydra_overloads import (
    MISSING,
    instantiate,
    load_from_yaml,
    save_as_yaml,
    to_yaml,
)
from ._launch import launch
from .structured_configs import (
    ZenField,
    builds,
    hydrated_dataclass,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
)
from .structured_configs._implementations import get_target
from .structured_configs._type_guards import is_partial_builds, uses_zen_processing
from .wrapper import ZenStore, store, zen

__all__ = [
    "builds",
    "hydrated_dataclass",
    "just",
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
