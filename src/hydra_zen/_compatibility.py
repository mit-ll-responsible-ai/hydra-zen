# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import Counter, deque
from enum import Enum
from functools import partial
from pathlib import Path, PosixPath, WindowsPath
from typing import NamedTuple, Set

import hydra
import omegaconf
from typing_extensions import Final

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


# OmegaConf issue 830 describes a bug associated with structured configs
# composed via inheritance, where the child's attribute is a default-factory
# and the parent's corresponding attribute is not.
# We provide downstream workarounds until an upstream fix is released.
#
# Uncomment dynamic setting once OmegaConf merges fix:
# https://github.com/omry/omegaconf/pull/832
PATCH_OMEGACONF_830: Final = OMEGACONF_VERSION < Version(2, 2, 1)

# Hydra's instantiate API now supports partial-instantiation, indicated
# by a `_partial_ = True` attribute.
# https://github.com/facebookresearch/hydra/pull/1905
HYDRA_SUPPORTS_PARTIAL: Final = Version(1, 1, 1) < HYDRA_VERSION

HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES: Final = OMEGACONF_VERSION >= Version(2, 2, 0)
HYDRA_SUPPORTS_BYTES: Final = OMEGACONF_VERSION >= Version(2, 2, 0)
HYDRA_SUPPORTS_Path: Final = OMEGACONF_VERSION >= Version(2, 2, 1)

# Indicates primitive types permitted in type-hints of structured configs
HYDRA_SUPPORTED_PRIMITIVE_TYPES: Final = {int, float, bool, str, Enum}
# Indicates types of primitive values permitted in configs
HYDRA_SUPPORTED_PRIMITIVES = {int, float, bool, str, list, tuple, dict, NoneType}
ZEN_SUPPORTED_PRIMITIVES: Set[type] = {
    set,
    frozenset,
    complex,
    partial,
    bytearray,
    deque,
    Counter,
    range,
}


if HYDRA_SUPPORTS_BYTES:  # pragma: no cover
    HYDRA_SUPPORTED_PRIMITIVES.add(bytes)
    HYDRA_SUPPORTED_PRIMITIVE_TYPES.add(bytes)
else:  # pragma: no cover
    ZEN_SUPPORTED_PRIMITIVES.add(bytes)

_path_types = {Path, PosixPath, WindowsPath}

if HYDRA_SUPPORTS_Path:  # pragma: no cover
    HYDRA_SUPPORTED_PRIMITIVES.update(_path_types)
    HYDRA_SUPPORTED_PRIMITIVE_TYPES.add(Path)
else:  # pragma: no cover
    ZEN_SUPPORTED_PRIMITIVES.update(_path_types)

del _path_types
