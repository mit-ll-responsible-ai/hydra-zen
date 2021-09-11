# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

from typing_extensions import Protocol, runtime_checkable

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
    "Importable",
]


_T = TypeVar("_T", covariant=True)
_T2 = TypeVar("_T2", covariant=False, contravariant=False)


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


class _Importable(Protocol):
    __module__: str
    __name__: str


Importable = TypeVar("Importable")


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]


@runtime_checkable
class Builds(DataClass, Protocol[_T]):  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        ...

    _target_: str


@runtime_checkable
class Just(Builds, Protocol[_T]):  # pragma: no cover
    path: str  # interpolated string for importing obj
    _target_: str = "hydra_zen.funcs.get_obj"


@runtime_checkable
class PartialBuilds(Builds, Protocol[_T2]):  # pragma: no cover
    _partial_target_: Just[_T2]
    _target_: str = "hydra_zen.funcs.partial"


@runtime_checkable
class HasTarget(Protocol):  # pragma: no cover
    _target_: str


@runtime_checkable
class HasPartialTarget(Protocol):  # pragma: no cover
    _partial_target_: Just
