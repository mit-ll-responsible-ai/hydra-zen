# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Simple helper functions used to implement `just` and `builds`. This module is designed specifically so
that these functions have a legible module-path when they appear in configuration files.
"""

import functools as _functools
import typing as _typing

from hydra._internal import utils as _hydra_internal_utils
from hydra.utils import log as _log

from hydra_zen.typing import Partial as _Partial

_T = _typing.TypeVar("_T")

__all__ = ["partial", "get_obj"]


def partial(
    *args: _typing.Any, _partial_target_: _typing.Callable[..., _T], **kwargs: _typing.Any
) -> _Partial[_T]:
    """Equivalent to ``functools.partial`` but provides a named parameter for the callable."""
    return _functools.partial(_partial_target_, *args, **kwargs)


def get_obj(*, path: str) -> _typing.Union[type, _typing.Callable[..., _typing.Any]]:
    """Imports an object given the specified path."""
    try:
        cl = _hydra_internal_utils._locate(path)
        return cl
    except Exception as e:  # pragma: no cover
        _log.error(f"Error getting callable at {path} : {e}")
        raise e
