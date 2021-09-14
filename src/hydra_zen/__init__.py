# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from ._hydra_overloads import (
    MISSING,
    instantiate,
    load_from_yaml,
    save_as_yaml,
    to_yaml,
)
from ._version import get_versions
from .structured_configs import builds, hydrated_dataclass, just, mutable_value
from .structured_configs._implementations import get_target

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
]

__version__ = get_versions()["version"]
del get_versions
