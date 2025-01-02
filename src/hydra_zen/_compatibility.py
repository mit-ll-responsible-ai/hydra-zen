# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import Counter, defaultdict, deque
from datetime import timedelta
from enum import Enum
from functools import partial
from pathlib import Path, PosixPath, WindowsPath
from typing import Final, NamedTuple

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
