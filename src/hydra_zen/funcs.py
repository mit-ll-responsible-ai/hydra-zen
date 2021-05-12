# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Simple helper functions used to implement `just` and `builds`. This module is designed specifically so
that these functions have a legible module-path when they appear in configuration files.
"""

import functools as _functools
import typing as _typing
from typing import Any, Callable, Union

from hydra._internal import utils as hydra_internal_utils
from hydra.utils import log

from hydra_zen.typing import Partial

_T = _typing.TypeVar("_T")

__all__ = ["partial", "get_obj"]


def partial(
    *args: Any, _partial_target_: Callable[..., _T], **kwargs: Any
) -> Partial[_T]:
    """Equivalent to ``functools.partial`` but provides a named parameter for the callable."""
    return _functools.partial(_partial_target_, *args, **kwargs)


def get_obj(*, path: str) -> Union[type, Callable[..., Any]]:
    """Imports an object given the specified path."""
    try:
        cl = hydra_internal_utils._locate(path)
        return cl
    except Exception as e:  # pragma: no cover
        log.error(f"Error getting callable at {path} : {e}")
        raise e
