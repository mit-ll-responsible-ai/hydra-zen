# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# pyright: strict

import sys
import types
from enum import Enum
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    TYPE_CHECKING,
    Any,
    ByteString,
    Callable,
    ClassVar,
    Dict,
    FrozenSet,
    Generic,
    List,
    Mapping,
    NewType,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from omegaconf import DictConfig, ListConfig
from typing_extensions import (
    Final,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypedDict,
    runtime_checkable,
)

__all__ = [
    "Just",
    "Builds",
    "PartialBuilds",
    "Partial",
    "Importable",
    "SupportedPrimitive",
    "ZenWrappers",
    "ZenPartialBuilds",
    "HydraPartialBuilds",
    "ZenConvert",
]

P = ParamSpec("P")
R = TypeVar("R")


class EmptyDict(TypedDict):
    pass


T = TypeVar("T", covariant=True)
T2 = TypeVar("T2")
T3 = TypeVar("T3")

T4 = TypeVar("T4", bound=Callable[..., Any])


InstOrType: TypeAlias = Union[T, Type[T]]


if TYPE_CHECKING:  # pragma: no cover
    from dataclasses import Field  # provided by typestub but not generic at runtime
else:

    class Field(Protocol[T2]):  # pragma: no cover
        name: str
        type: Type[T2]
        default: T2
        default_factory: Callable[[], T2]
        repr: bool
        hash: Optional[bool]
        init: bool
        compare: bool
        metadata: Mapping[str, Any]


@runtime_checkable
class Partial(Protocol[T2]):  # pragma: no cover
    @property
    def func(self) -> Callable[..., T2]:
        ...

    @property
    def args(self) -> Tuple[Any, ...]:
        ...

    @property
    def keywords(self) -> Dict[str, Any]:
        ...

    def __new__(
        cls: Type[Self], __func: Callable[..., T2], *args: Any, **kwargs: Any
    ) -> Self:
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> T2:
        ...

    if sys.version_info >= (3, 9):  # pragma: no cover

        def __class_getitem__(cls, item: Any) -> types.GenericAlias:
            ...


InterpStr = NewType("InterpStr", str)


class DataClass_(Protocol):  # pragma: no cover
    # doesn't provide __init__, __getattribute__, etc.
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]


class DataClass(DataClass_, Protocol):  # pragma: no cover
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...


@runtime_checkable
class Builds(DataClass, Protocol[T]):  # pragma: no cover
    _target_: ClassVar[str]


class BuildsWithSig(Builds[T], Protocol[T, P]):  # pragma: no cover
    def __init__(self, *args: P.args, **kwds: P.kwargs):  # pragma: no cover
        ...


@runtime_checkable
class Just(Builds[T], Protocol[T]):  # pragma: no cover
    path: ClassVar[str]  # interpolated string for importing obj
    _target_: ClassVar[Literal["hydra_zen.funcs.get_obj"]] = "hydra_zen.funcs.get_obj"


class ZenPartialMixin(Protocol[T]):  # pragma: no cover
    _zen_target: ClassVar[str]
    _zen_partial: ClassVar[Literal[True]] = True


class HydraPartialMixin(Protocol[T]):  # pragma: no cover
    _partial_: ClassVar[Literal[True]] = True


@runtime_checkable
class ZenPartialBuilds(Builds[T], ZenPartialMixin[T], Protocol[T]):  # pragma: no cover
    _target_: ClassVar[
        Literal["hydra_zen.funcs.zen_processing"]
    ] = "hydra_zen.funcs.zen_processing"


@runtime_checkable
class HydraPartialBuilds(
    Builds[T], HydraPartialMixin[T], Protocol[T]
):  # pragma: no cover
    ...


# Necessary, but not sufficient, check for PartialBuilds; useful for creating
# non-overlapping overloads
IsPartial: TypeAlias = Union[ZenPartialMixin[T], HydraPartialMixin[T]]

PartialBuilds: TypeAlias = Union[ZenPartialBuilds[T], HydraPartialBuilds[T]]


@runtime_checkable
class HasTarget(Protocol):  # pragma: no cover
    _target_: str


Importable = TypeVar("Importable", bound=Callable[..., Any])

_HydraPrimitive: TypeAlias = Union[
    bool, None, int, float, str, ByteString, Path, WindowsPath, PosixPath
]

_SupportedViaBuilds = Union[
    Partial[Any],
    range,
    Set[Any],
]

_SupportedPrimitive: TypeAlias = Union[
    _HydraPrimitive,
    ListConfig,
    DictConfig,
    Callable[..., Any],
    Enum,
    DataClass_,
    complex,
    _SupportedViaBuilds,
    EmptyDict,  # not covered by Mapping[..., ...]]
]

if TYPE_CHECKING:  # pragma: no cover
    SupportedPrimitive: TypeAlias = Union[
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
else:
    # cleans up annotations for REPLs
    SupportedPrimitive = TypeVar("SupportedPrimitive")


ZenWrapper: TypeAlias = Union[
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
    ZenWrappers: TypeAlias = Union[ZenWrapper[T4], Sequence[ZenWrapper[T4]]]
else:
    # cleans up annotations for REPLs
    class ZenWrappers(Generic[T2]):  # pragma: no cover
        pass


DefaultsList = List[
    Union[str, DataClass_, Mapping[str, Union[None, str, Sequence[str]]]]
]


# Lists all zen-convert settings and their types. Not part of public API
class AllConvert(TypedDict, total=True):
    dataclass: bool


# used for runtime type-checking
convert_types: Final = {"dataclass": bool}


class ZenConvert(TypedDict, total=False):
    """A TypedDict that provides a type-checked interface for specifying zen-convert
    options that configure the hydra-zen config-creation functions (e.g., `builds`,
    `just`, and `make_config`).

    Note that, at runtime, `ZenConvert` is simply a dictionary with type-annotations. There is no enforced runtime validation of its keys and values.

    Parameters
    ----------
    dataclass : bool
        If `True` any dataclass type/instance without a `_target_` field is
        automatically converted to a targeted config that will instantiate to that type/
        instance. Otherwise the dataclass type/instance will be passed through as-is.

        Note that this only works with statically-defined dataclass types, whereas
        :func:`~hydra_zen.make_config` and :py:func:`dataclasses.make_dataclass`
        dynamically generate dataclass types. Additionally, this feature is not
        compatible with a dataclass instance whose type possesses an `InitVar` field.

    Examples
    --------
    >>> from hydra_zen.typing import ZenConvert as zc
    >>> zc()
    {}
    >>> zc(dataclass=True)
    {"dataclass": True}
    >>> zc(apple=1)  # static type-checker will raise, but runtime will not
    {"apple": 1}

    **Configuring dataclass auto-config behaviors**

    >>> from hydra_zen import instantiate as I
    >>> from hydra_zen import builds, just
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class B:
    ...     x: int
    >>> b = B(x=1)

    >>> I(just(b))
    B(x=1)
    >>> I(just(b, zen_convert=zc(dataclass=False)))  # returns omegaconf.DictConfig
    {"x": 1}

    >>> I(builds(dict, y=b))
    {'y': B(x=1)}
    >>> I(builds(dict, y=b, zen_convert=zc(dataclass=False)))  # returns omegaconf.DictConfig
    {'y': {'x': 1}}

    >>> I(make_config(y=b))  # returns omegaconf.DictConfig
    {'y': {'x': 1}}
    >>> I(make_config(y=b, zen_convert=zc(dataclass=True, hydra_convert="all")))
    {'y': B(x=1)}

    Auto-config support does not work with dynamically-generated dataclass types

    >>> just(make_config(z=1))
    HydraZenUnsupportedPrimitiveError: ...
    >>> I(just(make_config(z=1), zen_convert=zc(dataclass=False)))
    {'z': 1}

    A dataclass with a `_target_` field will not be converted:

    >>> @dataclass
    ... class BuildsStr:
    ...     _target_: str = 'builtins.str'
    ...
    >>> BuildsStr is just(BuildsStr)
    True
    >>> (builds_str := BuildsStr()) is just(builds_str)
    True
    """

    dataclass: bool
