# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import Field, _DataclassParams
from typing import Any, Callable, Dict, Generic, Tuple, TypeVar

from typing_extensions import Literal, Protocol, runtime_checkable

__all__ = [
    "Importable",
    "DataClass",
    "Instantiable",
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
    # We are doing something a bit hacky here.. but instead of
    # using, e.g., Type[Builds] everywhere, it is nicer to use `Builds`.
    # Thus we add this __call__ method to make it look like objects
    # of type `Build` (which would be an *instance* of Build) are
    # instantiable
    # def __call__(self, *args, **kwargs) -> "DataClass":  # pragma: no cover
    #     ...

    __dataclass_fields__: Dict[str, Field]
    __dataclass_params__: _DataclassParams


class Instantiable(DataClass, Protocol[_T]):  # pragma: no cover
    _target_: str


@runtime_checkable
class Just(Instantiable, Protocol[_T]):
    path: str  # interpolated string for importing obj
    _target_: str = "hydra_utils.funcs.get_obj"


@runtime_checkable
class Builds(Instantiable, Protocol[_T]):  # pragma: no cover
    _convert_: Literal["none", "partial", "all"]
    _recursive_: bool


@runtime_checkable
class PartialBuilds(Builds, Protocol[_T]):
    _partial_target_: str
