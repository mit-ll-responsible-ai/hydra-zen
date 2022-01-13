# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Mapping,
    NewType,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from omegaconf import DictConfig, ListConfig
from typing_extensions import Protocol, TypedDict, runtime_checkable

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
    "Importable",
    "SupportedPrimitive",
]


class EmptyDict(TypedDict):
    pass


_T = TypeVar("_T", covariant=True)
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@runtime_checkable
class Partial(Protocol[T2]):
    func: Callable[..., T2]
    args: Tuple[Any, ...]
    keywords: Dict[str, Any]

    def __new__(
        cls: Type[T3], func: Callable[..., T2], *args: Any, **kwargs: Any
    ) -> T3:  # pragma: no cover
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> T2:  # pragma: no cover
        ...


InterpStr = NewType("InterpStr", str)


class _DataClass(Protocol):  # pragma: no cover
    # doesn't provide __init__, __getattribute__, etc.
    __dataclass_fields__: Dict[str, Field]


class DataClass(_DataClass, Protocol):  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        ...

    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
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
class HydraPartialBuilds(Builds, Protocol[_T]):  # pragma: no cover
    _partial_: bool = True


@runtime_checkable
class HasTarget(Protocol):  # pragma: no cover
    _target_: str


@runtime_checkable
class HasPartialTarget(Protocol):  # pragma: no cover
    _zen_partial: bool = True


Importable = TypeVar("Importable", bound=Callable)

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
    Partial,
    range,
    set,
    EmptyDict,  # not covered by Mapping[..., ...]
]

SupportedPrimitive = Union[
    _SupportedPrimitive,
    FrozenSet["SupportedPrimitive"],
    # Even thought this is redundant with Sequence, it seems to
    # be needed for pyright to do proper checking of tuple contents
    Tuple["SupportedPrimitive"],
    # Mutable generic containers need to be invariant, so
    # we have to settle for Sequence/Mapping. While this
    # is overly permissive in terms of sequence-type, it
    # at least affords quality checking of sequence content
    Sequence["SupportedPrimitive"],
    # Mapping is covariant only in value
    Mapping[Any, "SupportedPrimitive"],
]
