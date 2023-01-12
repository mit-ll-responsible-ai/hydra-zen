# Copyright (c) 2023 Massachusetts Institute of Technology
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
    Required,
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


if TYPE_CHECKING:
    from dataclasses import Field  # provided by typestub but not generic at runtime
else:

    class Field(Protocol[T2]):
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
class Partial(Protocol[T2]):
    __call__: Callable[..., T2]

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

    if TYPE_CHECKING and sys.version_info >= (3, 9):  # pragma: no cover

        def __class_getitem__(cls, item: Any) -> types.GenericAlias:
            ...


InterpStr = NewType("InterpStr", str)


class DataClass_(Protocol):
    # doesn't provide __init__, __getattribute__, etc.
    __dataclass_fields__: ClassVar[Dict[str, Field[Any]]]


class DataClass(DataClass_, Protocol):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    def __getattribute__(self, __name: str) -> Any:
        ...

    def __setattr__(self, __name: str, __value: Any) -> None:
        ...


@runtime_checkable
class HasTarget(Protocol):
    _target_: str


@runtime_checkable
class Builds(DataClass, Protocol[T]):
    _target_: ClassVar[str]


class BuildsWithSig(Builds[T], Protocol[T, P]):
    def __init__(self, *args: P.args, **kwds: P.kwargs):
        ...


@runtime_checkable
class Just(Builds[T], Protocol[T]):
    path: str  # interpolated string for importing obj
    _target_: ClassVar[Literal["hydra_zen.funcs.get_obj"]] = "hydra_zen.funcs.get_obj"


class ZenPartialMixin(Protocol[T]):
    _zen_target: ClassVar[str]
    _zen_partial: ClassVar[Literal[True]] = True


class HydraPartialMixin(Protocol[T]):
    _partial_: ClassVar[Literal[True]] = True


@runtime_checkable
class ZenPartialBuilds(Builds[T], ZenPartialMixin[T], Protocol[T]):
    _target_: ClassVar[
        Literal["hydra_zen.funcs.zen_processing"]
    ] = "hydra_zen.funcs.zen_processing"


@runtime_checkable
class HydraPartialBuilds(Builds[T], HydraPartialMixin[T], Protocol[T]):
    ...


# Necessary, but not sufficient, check for PartialBuilds; useful for creating
# non-overlapping overloads
IsPartial: TypeAlias = Union[ZenPartialMixin[T], HydraPartialMixin[T]]

PartialBuilds: TypeAlias = Union[ZenPartialBuilds[T], HydraPartialBuilds[T]]


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

if TYPE_CHECKING:
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
if TYPE_CHECKING:
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

GroupName: TypeAlias = Optional[str]
NodeName: TypeAlias = str
Node: TypeAlias = Any


# TODO: make immutable
class StoreEntry(TypedDict):
    name: NodeName
    group: GroupName
    package: Optional[str]
    provider: Optional[str]
    node: Node


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
    >>> # static type-checker will raise, but runtime will not
    >>> zc(apple=1)  # type: ignore
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
    >>> I(make_config(y=b, zen_convert=zc(dataclass=True), hydra_convert="all"))
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


class _AllPyDataclassOptions(TypedDict, total=False):
    cls_name: str
    namespace: Optional[Dict[str, Any]]
    bases: Tuple[Type[DataClass_], ...]
    init: bool
    repr: bool
    eq: bool
    order: bool
    unsafe_hash: bool
    frozen: bool


class _Py310Dataclass(_AllPyDataclassOptions, total=False):
    # py310+
    match_args: bool
    kw_only: bool
    slots: bool


class _Py311Dataclass(_Py310Dataclass, total=False):
    weakref_slot: bool


if sys.version_info < (3, 10):
    _StrictDataclassOptions = _AllPyDataclassOptions
elif sys.version_info < (3, 11):
    _StrictDataclassOptions = _Py310Dataclass
else:  # pragma: no cover
    _StrictDataclassOptions = _Py311Dataclass


class StrictDataclassOptions(_StrictDataclassOptions):
    cls_name: Required[str]


class DataclassOptions(_Py311Dataclass, total=False):
    """Specifies dataclass-creation options via `builds`, `just` et al.

    Note that, unlike :func:`dataclasses.make_dataclass`, the default value for
    `unsafe_hash` is `True` for hydra-zen's dataclass-generating functions.
    See the documentation for :func:`dataclasses.make_dataclass` for more details [1]_.

    Options that are not supported by the local Python version will be ignored by
    hydra-zen's config-creation functions.

    Parameters
    ----------
    cls_name : str, optional
        If specified, determines the name of the returned class object. Otherwise the
        name is inferred by hydra-zen.

    module : str, default='typing'
        If specified, sets the `__module__` attribute of the resulting dataclass.

        Specifying the module string-path in which the dataclass was generated, and
        specifying `cls_name` as the symbol that references the dataclass, will enable
        pickle-compatibility for that dataclass. See the Examples section for
        clarification.

        This is a hydra-zen exclusive feature.

    init : bool, optional (default=True)
        If true (the default), a __init__() method will be generated. If the class
        already defines __init__(), this parameter is ignored.

    repr : bool, optional (default=True)
        If true (the default), a `__repr__()` method will be generated. The generated
        repr string will have the class name and the name and repr of each field, in
        the order they are defined in the class. Fields that are marked as being
        excluded from the repr are not included. For example:
        `InventoryItem(name='widget', unit_price=3.0, quantity_on_hand=10)`.

    eq : bool, optional (default=True)
        If true (the default), an __eq__() method will be generated. This method
        compares the class as if it were a tuple of its fields, in order. Both
        instances in the comparison must be of the identical type.

    order : bool, optional (default=False)
        If true (the default is `False`), `__lt__()`, `__le__()`, `__gt__()`, and
        `__ge__()` methods will be generated. These compare the class as if it were a
        tuple of its fields, in order. Both instances in the comparison must be of the
        identical type. If order is true and eq is false, a ValueError is raised.

        If the class already defines any of `__lt__()`, `__le__()`, `__gt__()`, or
        `__ge__()`, then `TypeError` is raised.

    unsafe_hash : bool, optional (default=False)
        If `False` (the default), a `__hash__()` method is generated according to how
        `eq` and `frozen` are set.

        If `eq` and `frozen` are both true, by default `dataclass()` will generate a
        `__hash__()` method for you. If `eq` is true and `frozen` is false, `__hash__()
        ` will be set to `None`, marking it unhashable. If `eq` is false, `__hash__()`
        will be left untouched meaning the `__hash__()` method of the superclass will
        be used (if the superclass is object, this means it will fall back to id-based
        hashing).

    frozen : bool, optional (default=False)
        If true (the default is `False`), assigning to fields will generate an
        exception. This emulates read-only frozen instances.

    match_args : bool, optional (default=True)
        (*New in version 3.10*) If true (the default is `True`), the `__match_args__`
        tuple will be created from the list of parameters to the generated `__init__()`
        method (even if `__init__()` is not generated, see above). If false, or if
        `__match_args__` is already defined in the class, then `__match_args__` will
        not be generated.

    kw_only : bool, optional (default=False)
        (*New in version 3.10*) If true (the default value is `False`), then all fields
        will be marked as keyword-only.

    slots : bool, optional (default=False)
        (*New in version 3.10*) If true (the default is `False`), `__slots__` attribute
        will be generated and new class will be returned instead of the original one.
        If `__slots__` is already defined in the class, then `TypeError` is raised.

    weakref_slot : bool, optional (default=False)
        (*New in version 3.11*) If true (the default is `False`), add a slot named
        “__weakref__”, which is required to make an instance weakref-able. It is an
        error to specify `weakref_slot=True` without also specifying `slots=True`.

    References
    ----------
    .. [1] https://docs.python.org/3/library/dataclasses.html
    .. [2] https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    Notes
    -----
    This is a typed dictionary, which provides static type information (e.g. type
    checking and auto completion options) to tooling. Note, however, that it provides
    no runtime checking of its keys and values.

    Examples
    --------
    >>> from hydra_zen.typing import DataclassOptions as Opts
    >>> from hydra_zen import builds, make_config, make_custom_builds_fn

    Creating a frozen config.

    >>> conf = make_config(x=1, zen_dataclass=Opts(frozen=True))()
    >>> conf.x = 2
    FrozenInstanceError: cannot assign to field 'x'

    Creating a pickle-compatible config:

    The dynamically-generated classes created by `builds`, `make_config`, and `just`
    can be made pickle-compatible by specifying the name of the symbol that it is
    assigned to and the module in which it was defined.

    .. code-block:: python

       # contents of mylib/foo.py
       from pickle import dumps, loads
       from hydra_zen import builds

       DictConf = builds(dict,
                         zen_dataclass={'module': 'mylib.foo',
                                        'cls_name': 'DictConf'})

       assert DictConf is loads(dumps(DictConf))

    Using namespace to add a method to a config instance.

    >>> conf = make_config(
    ...     x=100,
    ...     zen_dataclass=Opts(
    ...         namespace=dict(add_x=lambda self, y: self.x + y),
    ...     ),
    ... )()
    >>> conf.add_x(2)
    102

    Dataclasse objects created by hydra-zen's config-creation functions will be created
    with `unsafe_hash=True` by default. This is in contrast with the default behavior of
    :py:func:`dataclasses.dataclass`. This is to help ensure smooth compatibility
    through Python 3.11, which changed the mutability checking rules for dataclasses
    [2]_.

    >>> from dataclasses import make_dataclass
    >>> DataClass = make_dataclass(fields=[], cls_name="A")
    >>> DataClass.__hash__
    None

    >>> Conf = make_config(x=2)
    >>> Conf.__hash__
    <function types.__create_fn__.<locals>.__hash__(self)>

    >>> UnHashConf = make_config(x=2, zen_dataclass=Opts(unsafe_hash=False))
    >>> UnHashConf.__hash__
    None
    """

    module: str  # zen-only


def _permitted_keys(typed_dict: Any) -> FrozenSet[str]:
    return typed_dict.__required_keys__ | typed_dict.__optional_keys__


DEFAULT_DATACLASS_OPTIONS = DataclassOptions(unsafe_hash=True)
PERMITTED_DATACLASS_OPTIONS = _permitted_keys(DataclassOptions)
UNSUPPORTED_DATACLASS_OPTIONS = _permitted_keys(_Py311Dataclass) - _permitted_keys(
    StrictDataclassOptions
)
del _AllPyDataclassOptions, _Py310Dataclass, _Py311Dataclass
