# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field, _DataclassParams
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

from typing_extensions import Literal, Protocol, runtime_checkable

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
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


class _Importable(Protocol):
    __module__: str
    __name__: str


Importable = TypeVar("Importable")


class DataClass(Protocol):
    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


@runtime_checkable
class Builds(DataClass, Protocol[_T]):  # pragma: no cover
    def __init__(self, *args, **kwargs) -> None:
        ...

    _target_: str


@runtime_checkable
class Just(Builds, Protocol[_T]):
    path: str  # interpolated string for importing obj
    _target_: str = "hydra_utils.funcs.get_obj"


@runtime_checkable
class PartialBuilds(Builds, Protocol[_T]):
    _partial_target_: str
