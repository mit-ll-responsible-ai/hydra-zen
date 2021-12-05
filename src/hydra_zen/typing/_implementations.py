# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    NewType,
    Tuple,
    TypeVar,
    Union,
)

from omegaconf import DictConfig, ListConfig
from typing_extensions import Protocol, runtime_checkable

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
    "Importable",
    "SupportedPrimitive",
]


_T = TypeVar("_T", covariant=True)


class Partial(Generic[_T]):
    func: Callable[..., _T]
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

    def __init__(
        self, func: Callable[..., _T], *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> _T:  # pragma: no cover
        ...


InterpStr = NewType("InterpStr", str)

Importable = TypeVar("Importable")


class _DataClass(Protocol):  # pragma: no cover
    # doesn't provide __init__, __getattribute__, etc.
    __dataclass_fields__: Dict[str, Field]


class DataClass(_DataClass, Protocol):  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        ...

    def __getattribute__(self, name: str) -> Any:
        ...

    def __setattr__(self, name: str, value: Any) -> None:
        ...


@runtime_checkable
class Builds(DataClass, Protocol[_T]):  # pragma: no cover

    _target_: str


@runtime_checkable
class Just(Builds, Protocol[_T]):  # pragma: no cover
    path: str  # interpolated string for importing obj
    _target_: str = "hydra_zen.funcs.get_obj"


@runtime_checkable
class PartialBuilds(Builds, Protocol[_T]):  # pragma: no cover
    _target_: str = "hydra_zen.funcs.zen_processing"
    _zen_target: str
    _zen_partial: bool = True


@runtime_checkable
class HasTarget(Protocol):  # pragma: no cover
    _target_: str


@runtime_checkable
class HasPartialTarget(Protocol):  # pragma: no cover
    _zen_partial: bool = True


_HydraPrimitive = Union[
    bool,
    None,
    int,
    float,
    str,
]

_SupportedPrimitive = Union[
    _HydraPrimitive,
    ListConfig,
    DictConfig,
    type,
    Callable,
    Enum,
    _DataClass,
    complex,
    Path,
    range,
    # Very sadly, mutable generic containers need to be invariant.
    # See: https://mypy.readthedocs.io/en/stable/common_issues.html#invariance-vs-covariance
    #
    # So we can't do e.g. List['SupportedPrimitive'] for accurate, recursive
    # lists. We would need to so List[int] | List[str] | ... ad naeseum
    # in order to avoide hitting users with false-positives.
    # But hey! At least our frozen-sets are very accurately typed :P
    list,
    dict,
    set,
    Deque,
]

SupportedPrimitive = Union[
    _SupportedPrimitive,
    FrozenSet["SupportedPrimitive"],
    Tuple["SupportedPrimitive", ...],
]
