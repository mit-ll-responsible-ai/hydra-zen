# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
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
from typing_extensions import Literal, Protocol, TypedDict, runtime_checkable

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
    "Importable",
    "SupportedPrimitive",
    "ZenWrappers",
]


class EmptyDict(TypedDict):
    pass


T = TypeVar("T", covariant=True)
T2 = TypeVar("T2")
T3 = TypeVar("T3")

T4 = TypeVar("T4", bound=Callable)


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
    __dataclass_fields__: ClassVar[Dict[str, Field]]


class DataClass(_DataClass, Protocol):  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        ...

    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...


@runtime_checkable
class Builds(DataClass, Protocol[T]):  # pragma: no cover
    _target_: ClassVar[str]


@runtime_checkable
class Just(Builds, Protocol[T]):  # pragma: no cover
    path: ClassVar[str]  # interpolated string for importing obj
    _target_: ClassVar[Literal["hydra_zen.funcs.get_obj"]] = "hydra_zen.funcs.get_obj"


@runtime_checkable
class PartialBuilds(Builds, Protocol[T]):  # pragma: no cover
    _target_: ClassVar[
        Literal["hydra_zen.funcs.zen_processing"]
    ] = "hydra_zen.funcs.zen_processing"
    _zen_target: ClassVar[str]
    _zen_partial: ClassVar[Literal[True]] = True


@runtime_checkable
class HydraPartialBuilds(Builds, Protocol[T]):  # pragma: no cover
    _partial_: ClassVar[Literal[True]] = True


@runtime_checkable
class HasTarget(Protocol):  # pragma: no cover
    _target_: str


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
    Tuple["SupportedPrimitive", ...],
    # Mutable generic containers need to be invariant, so
    # we have to settle for Sequence/Mapping. While this
    # is overly permissive in terms of sequence-type, it
    # at least affords quality checking of sequence content
    Sequence["SupportedPrimitive"],
    # Mapping is covariant only in value
    Mapping[Any, "SupportedPrimitive"],
]


ZenWrapper = Union[
    None,
    Builds[Callable[[T4], T4]],
    PartialBuilds[Callable[[T4], T4]],
    Just[Callable[[T4], T4]],
    Type[Builds[Callable[[T4], T4]]],
    Type[PartialBuilds[Callable[[T4], T4]]],
    Type[Just[Callable[[T4], T4]]],
    Callable[[T4], T4],
    str,
]
if TYPE_CHECKING:  # pragma: no cover
    ZenWrappers = Union[ZenWrapper, Sequence[ZenWrapper]]
else:
    ZenWrappers = TypeVar("ZenWrappers")
