# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Simple helper functions used to implement `just` and `builds`. This module is designed specifically so
that these functions have a legible module-path when they appear in configuration files.
"""
import functools as _functools
import typing as _tp

from hydra._internal import utils as _hydra_internal_utils
from hydra.utils import log as _log

from hydra_zen.structured_configs._utils import (
    is_interpolated_string as _is_interpolated_string,
)
from hydra_zen.typing import Partial as _Partial

__all__ = ["partial", "get_obj", "zen_processing"]

_T = _tp.TypeVar("_T")

_Wrapper = _tp.Callable[[_tp.Callable[..., _tp.Any]], _tp.Callable[..., _tp.Any]]
_WrapperConf = _tp.Union[str, _Wrapper]


def partial(
    *args: _tp.Any,
    _partial_target_: _tp.Callable[..., _T],
    **kwargs: _tp.Any,
) -> _Partial[_T]:
    """Equivalent to ``functools.partial`` but provides a named parameter for the callable."""
    return _tp.cast(_Partial[_T], _functools.partial(_partial_target_, *args, **kwargs))


def get_obj(*, path: str) -> _tp.Union[type, _tp.Callable[..., _tp.Any]]:
    """Imports an object given the specified path."""
    try:
        cl = _hydra_internal_utils._locate(path)
        return cl
    except Exception as e:  # pragma: no cover
        _log.error(f"Error getting callable at {path} : {e}")
        raise e


def zen_processing(
    *args: _tp.Any,
    _zen_target: str,
    _zen_partial: bool = False,
    _zen_exclude: _tp.Sequence[str] = tuple(),
    _zen_wrappers: _tp.Union[_WrapperConf, _tp.Sequence[_WrapperConf]] = tuple(),
    **kwargs: _tp.Any,
) -> _tp.Any:
    if isinstance(_zen_wrappers, str) or not isinstance(_zen_wrappers, _tp.Sequence):
        unresolved_wrappers: _tp.Sequence[_WrapperConf] = (_zen_wrappers,)
    else:
        unresolved_wrappers: _tp.Sequence[_WrapperConf] = _zen_wrappers
    del _zen_wrappers

    resolved_wrappers: _tp.List[_Wrapper] = []

    for _unresolved in unresolved_wrappers:
        if _unresolved is None:
            # We permit interpolated fields to resolve to `None`; this is
            # a nice pattern for enabling people to ergonomically toggle
            # wrappers off.
            continue
        if isinstance(_unresolved, str):
            # Hydra will have already raised on missing interpolation
            # keys by here
            assert not _is_interpolated_string(_unresolved)
            _unresolved = get_obj(path=_unresolved)

        if not callable(_unresolved):
            raise TypeError(
                f"Instantiating {_zen_target}: `zen_wrappers` was passed a non-callable object: {_unresolved}"
            )
        else:
            resolved = _unresolved
        del _unresolved
        resolved_wrappers.append(resolved)

    obj = get_obj(path=_zen_target)

    # first wrapper listed should be called first
    # [f1, f2, f3, ...] ->
    #    target = f1(target)
    #    target = f2(target)
    #    target = f3(target)
    #    ...
    for wrapper in resolved_wrappers:
        obj = wrapper(obj)

    if _zen_exclude:
        excluded_set = set(_zen_exclude)
        kwargs = {k: v for k, v in kwargs.items() if k not in excluded_set}

    if _zen_partial is True:
        return _functools.partial(obj, *args, **kwargs)
    return obj(*args, **kwargs)
