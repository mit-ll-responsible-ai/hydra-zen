# Copyright (c) 2021 Massachusetts Institute of Technology

"""
Simple helper functions used to implement `just` and `builds`. This module is designed specifically so
that these functions have a legible module-path when they appear in configuration files.
"""

import functools as _functools
import typing as _typing

_T = _typing.TypeVar("_T")

__all__ = ["partial", "identity"]


def partial(_partial_target_: _typing.Callable, *args, **kwargs) -> _typing.Callable:
    """Equivalent to ``functools.partial`` but provides a named parameter for the callable."""
    return _functools.partial(_partial_target_, *args, **kwargs)


def identity(obj: _T) -> _T:
    """Returns ``obj`` unchanged."""
    return obj
