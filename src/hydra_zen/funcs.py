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

__all__ = ["partial", "get_obj", "zen_processing"]

_T = _typing.TypeVar("_T")


def partial(
    *args: _typing.Any,
    _partial_target_: _typing.Callable[..., _T],
    **kwargs: _typing.Any,
) -> _Partial[_T]:
    """Equivalent to ``functools.partial`` but provides a named parameter for the callable."""
    return _typing.cast(
        _Partial[_T], _functools.partial(_partial_target_, *args, **kwargs)
    )


def get_obj(*, path: str) -> _typing.Union[type, _typing.Callable[..., _typing.Any]]:
    """Imports an object given the specified path."""
    try:
        cl = _hydra_internal_utils._locate(path)
        return cl
    except Exception as e:  # pragma: no cover
        _log.error(f"Error getting callable at {path} : {e}")
        raise e


def zen_processing(
    *args,
    _zen_target: str,
    _zen_partial: bool = False,
    _zen_exclude: _typing.Sequence[str] = tuple(),
    _zen_wrappers: _typing.Union[str, _typing.Sequence[str]] = tuple(),
    **kwargs,
):
    if isinstance(_zen_wrappers, str):
        _zen_wrappers = (_zen_wrappers,)

    # flip order: first wrapper listed should be called first
    wrappers = tuple(get_obj(path=z) for z in _zen_wrappers)[::-1]

    obj = get_obj(path=_zen_target)

    for wrapper in wrappers:
        obj = wrapper(obj)

    if _zen_exclude:
        excluded_set = set(_zen_exclude)
        kwargs = {k: v for k, v in kwargs.items() if k not in excluded_set}

    if _zen_partial is True:
        return _functools.partial(obj, *args, **kwargs)
    return obj(*args, **kwargs)
