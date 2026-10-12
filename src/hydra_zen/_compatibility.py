# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
import importlib
from collections import Counter, defaultdict, deque
from datetime import timedelta
from enum import Enum
from functools import partial
from pathlib import Path, PosixPath, WindowsPath
from types import ModuleType
from typing import Final, NamedTuple, Optional

import hydra
import omegaconf

NoneType = type(None)


class Version(NamedTuple):
    major: int
    minor: int
    patch: int


def _get_version(ver_str: str) -> Version:
    # Not for general use. Tested only for Hydra and OmegaConf
    # version string styles

    splits = ver_str.split(".")[:3]
    if not len(splits) == 3:  # pragma: no cover
        raise ValueError(f"Version string {ver_str} couldn't be parsed")

    major, minor = (int(v) for v in splits[:2])
    patch_str, *_ = splits[-1].split("rc")

    return Version(major=major, minor=minor, patch=int(patch_str))


OMEGACONF_VERSION: Final = _get_version(omegaconf.__version__)
HYDRA_VERSION: Final = _get_version(hydra.__version__)


# Indicates primitive types permitted in type-hints of structured configs
HYDRA_SUPPORTED_PRIMITIVE_TYPES: Final = frozenset(
    {int, float, bool, str, Enum, bytes, Path}
)
# Indicates types of primitive values permitted in configs
HYDRA_SUPPORTED_PRIMITIVES = frozenset(
    {
        int,
        float,
        bool,
        str,
        list,
        tuple,
        dict,
        NoneType,
        bytes,
        Path,
        PosixPath,
        WindowsPath,
    }
)
ZEN_SUPPORTED_PRIMITIVES: frozenset[type] = frozenset(
    {
        set,
        frozenset,
        complex,
        partial,
        bytearray,
        deque,
        Counter,
        range,
        timedelta,
        defaultdict,
    }
)


HYDRA_SUPPORTS_OBJECT_CONVERT = HYDRA_VERSION >= Version(1, 3, 0)

# OmegaConf 2.4.0 (PR #1296) teaches its yaml loader to construct
# pathlib._local.* paths (which Python 3.13 emits for pathlib.Path et al.).
# On older OmegaConf we monkey-patch those constructors in ourselves.
OMEGACONF_HANDLES_PATHLIB_LOCAL = OMEGACONF_VERSION >= Version(2, 4, 0)


# Hydra 1.3.6 added a policy that authorizes targets during instantiation
# (`hydra._internal.target_policy`), which Hydra 1.4 reworked into an execution
# policy (`hydra._internal.execution_policy`). We feature-detect these rather
# than compare versions because Hydra 1.4 dev-builds report the same version.
#
# The module is deliberately never stored as an attribute of a hydra-zen module:
# an attribute path to Hydra's private policy state would be reachable from configs.
@functools.lru_cache(maxsize=None)
def _target_policy_module() -> Optional[ModuleType]:  # pragma: no cover
    """Returns Hydra's private target/execution policy module, if it exists."""
    for name in ("execution_policy", "target_policy"):
        try:
            return importlib.import_module(f"hydra._internal.{name}")
        except ImportError:
            pass
    return None


HYDRA_HAS_TARGET_POLICY: Final = _target_policy_module() is not None
HYDRA_HAS_EXECUTION_POLICY: Final = (
    getattr(_target_policy_module(), "__name__", None)
    == "hydra._internal.execution_policy"
)
