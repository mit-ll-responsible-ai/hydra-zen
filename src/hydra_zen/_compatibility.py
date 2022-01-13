# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import Counter, deque
from enum import Enum
from functools import partial
from pathlib import Path, PosixPath, WindowsPath
from typing import NamedTuple, Optional, Set

import hydra
import omegaconf
from typing_extensions import Final

NoneType = type(None)


class Version(NamedTuple):
    major: int
    minor: int
    patch: Optional[int] = None


def _get_version(ver_str: str) -> Version:
    # Not for general use. Tested only for Hydra and OmegaConf
    # version string styles
    major, minor = (int(v) for v in ver_str.split(".")[:2])
    return Version(major=major, minor=minor)


OMEGACONF_VERSION: Final = _get_version(omegaconf.__version__)
HYDRA_VERSION: Final = _get_version(hydra.__version__)


# OmegaConf issue 830 describes a bug associated with structured configs
# composed via inheritance, where the child's attribute is a default-factory
# and the parent's corresponding attribute is not.
# We provide downstream workarounds until an upstream fix is released.
#
# Uncomment dynamic setting once OmegaConf merges fix:
# https://github.com/omry/omegaconf/pull/832
PATCH_OMEGACONF_830: Final = True  # OMEGACONF_VERSION <= Version(2, 1)

# Hydra's instantiate API now supports partial-instantiation, indicated
# by a `_partial_ = True` attribute.
# https://github.com/facebookresearch/hydra/pull/1905
#
# Uncomment dynamice setting of `HYDRA_SUPPORTS_PARTIAL` once we can
# begin testing against nightly builds of Hydra
HYDRA_SUPPORTS_PARTIAL: Final = False  # Version(1, 1) < HYDRA_VERSION

# Indicates primitive types permitted in type-hints of structured configs
HYDRA_SUPPORTED_PRIMITIVE_TYPES: Final = {int, float, bool, str, Enum}
# Indicates types of primitive values permitted in configs
HYDRA_SUPPORTED_PRIMITIVES = {int, float, bool, str, list, tuple, dict, NoneType}
ZEN_SUPPORTED_PRIMITIVES: Set[type] = {
    set,
    frozenset,
    complex,
    partial,
    Path,
    PosixPath,
    WindowsPath,
    bytes,
    bytearray,
    deque,
    Counter,
    range,
}
