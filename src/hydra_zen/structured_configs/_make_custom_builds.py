# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
import inspect
import warnings
from collections.abc import Collection, Mapping
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Final,
    Optional,
    TypeVar,
    Union,
    cast,
    overload,
)

from typing_extensions import Literal

from hydra_zen.errors import HydraZenDeprecationWarning
from hydra_zen.typing import DataclassOptions, ZenWrappers
from hydra_zen.typing._builds_overloads import FullBuilds, PBuilds, StdBuilds
from hydra_zen.typing._implementations import InstOrType, ZenConvert

from ._implementations import BuildsFn, DefaultBuilds, builds
from ._utils import parse_dataclass_options

__all__ = ["make_custom_builds_fn"]


_builds_sig = inspect.signature(builds)
__BUILDS_DEFAULTS: Final[dict[str, Any]] = {
    name: p.default
    for name, p in _builds_sig.parameters.items()
    if p.kind is p.KEYWORD_ONLY
}
__BUILDS_DEFAULTS["zen_exclude"] = frozenset()
# TODO: Remove deprecated options once they are phased out
__BUILDS_DEFAULTS["frozen"] = False
__BUILDS_DEFAULTS["dataclass_name"] = None
del _builds_sig

# TODO: parameterize the return types and attach this
#       as a classmethod to `BuildsFn`

T = TypeVar("T")


# partial=False, pop-sig=True
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    populate_full_signature: Literal[True],
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> FullBuilds[T]: ...


# partial=True, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[True],
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    populate_full_signature: bool = ...,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> PBuilds[T]: ...


# partial=False, pop-sig=False
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: Literal[False] = ...,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> StdBuilds[T]: ...


# partial=False, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: bool,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> Union[FullBuilds[T], StdBuilds[T]]: ...


# partial=bool, pop-sig=False
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Union[bool, None],
    populate_full_signature: Literal[False] = ...,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> Union[PBuilds[T], StdBuilds[T]]: ...


# partial=bool, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Union[bool, None],
    populate_full_signature: bool,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> Union[
    FullBuilds[T],
    PBuilds[T],
    StdBuilds[T],
]: ...


def make_custom_builds_fn(
    *,
    zen_partial: Optional[bool] = None,
    populate_full_signature: bool = False,
    zen_exclude: Union[Collection[str], Callable[[str], bool]] = frozenset(),
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    zen_dataclass: Optional[DataclassOptions] = None,
    frozen: bool = False,
    zen_convert: Optional[ZenConvert] = None,
    builds_fn: InstOrType[BuildsFn[T]] = DefaultBuilds,
) -> Union[
    FullBuilds[T],
    PBuilds[T],
    StdBuilds[T],
]:
    """Returns the `builds` function, but with customized default values.

    E.g. ``make_custom_builds_fn(hydra_convert='all')`` will return a version
    of the `builds` function where the default value for ``hydra_convert``
    is ``'all'`` instead of ``None``.

    Parameters
    ----------
    zen_partial : bool, optional (default=False)
        Specifies a new the default value for ``builds(..., zen_partial=<..>)``

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        Specifies a new the default value for ``builds(..., zen_wrappers=<..>)``

    zen_meta : Optional[Mapping[str, Any]]
        Specifies a new the default value for ``builds(..., zen_meta=<..>)``

    populate_full_signature : bool, optional (default=False)
        Specifies a new the default value for ``builds(..., populate_full_signature=<..>)``

    zen_exclude : Collection[str] | Callable[[str], bool], optional (default=[])
        Specifies parameter names, or a function for checking names, to exclude
        those parameters from the config-creation process.

        Note that inherited fields cannot be excluded.

    zen_convert : Optional[ZenConvert]
        A dictionary that modifies hydra-zen's value and type conversion behavior.
        Consists of the following optional key-value pairs (:ref:`zen-convert`):

        - `dataclass` : `bool` (default=True):
            If `True` any dataclass type/instance without a
            `_target_` field is automatically converted to a targeted config
            that will instantiate to that type/instance. Otherwise the dataclass
            type/instance will be passed through as-is.

    zen_dataclass : Optional[DataclassOptions]
        A dictionary can specify any option that is supported by
        :py:func:`dataclasses.make_dataclass` other than `fields`.
        The default value for `unsafe_hash` is `True`.

        Additionally, the `module` field can be specified to enable pickle
        compatibility. See `hydra_zen.typing.DataclassOptions` for details.

    hydra_recursive : Optional[bool], optional (default=True)
        Specifies a new the default value for ``builds(..., hydra_recursive=<..>)``

    hydra_convert : Optional[Literal["none", "partial", "all", "object"]], optional (default="none")
        Specifies a new the default value for ``builds(..., hydra_convert=<..>)``

    frozen : bool, optional (default=False)
        .. deprecated:: 0.9.0
            `frozen` will be removed in hydra-zen 0.10.0. It is replaced by
            `zen_dataclass={'frozen': <bool>}`.

        Specifies a new the default value for ``builds(..., frozen=<..>)``

    builds_fn: BuildsFn[T]
        The builds-function whose defaults are modified.

    Returns
    -------
    custom_builds[T]
        The function `builds`, but with customized default values.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.

    Examples
    --------
    >>> from hydra_zen import builds, make_custom_builds_fn, instantiate

    **Basic usage**

    The following will create a `builds` function whose default value
    for ``zen_partial`` has been set to ``True``.

    >>> pbuilds = make_custom_builds_fn(zen_partial=True)

    I.e. using ``pbuilds(...)`` is equivalent to using
    ``builds(..., zen_partial=True)``.

    >>> instantiate(pbuilds(int))  # calls `functools.partial(int)`
    functools.partial(<class 'int'>)

    You can still specify ``zen_partial`` on a per-case basis with ``pbuilds``.

    >>> instantiate(pbuilds(int, zen_partial=False))  # calls `int()`
    0

    **Adding data validation to configs**

    Suppose that we want to enable runtime type-checking - using beartype -
    whenever our configs are being instantiated; then the following settings
    for `builds` would be handy.

    >>> # Note: beartype must be installed to use this feature
    >>> from hydra_zen.third_party.beartype import validates_with_beartype
    >>> build_a_bear = make_custom_builds_fn(
    ...     populate_full_signature=True,
    ...     hydra_convert="all",
    ...     zen_wrappers=validates_with_beartype,
    ... )

    Now all configs produced via ``build_a_bear`` will include type-checking
    during instantiation.

    >>> from typing_extensions import Literal
    >>> def f(x: Literal["a", "b"]): return x

    >>> Conf = build_a_bear(f)  # a conf that includes `validates_with_beartype`

    >>> instantiate(Conf, x="a")  # satisfies annotation: Literal["a", "b"]
    "a"

    >>> instantiate(Conf, x="c")  # violates annotation: Literal["a", "b"]
    <Validation error: "c" is not "a" or "b">
    """
    excluded_fields = frozenset({"dataclass_name", "hydra_defaults", "builds_bases"})
    fn = builds_fn.builds
    del builds_fn
    LOCALS = locals()
    # Ensures that new defaults added to `builds` must be reflected
    # in the signature of `make_custom_builds_fn`.
    assert (set(__BUILDS_DEFAULTS) - excluded_fields) <= set(LOCALS)

    _new_defaults = {
        name: LOCALS[name] for name in __BUILDS_DEFAULTS if name not in excluded_fields
    }

    _frozen = _new_defaults.pop("frozen")

    # This self-check was causing issues with type-checkers (pyright) -
    # causing them to resolve the return type of this function as "unknown"
    if not TYPE_CHECKING:  # pragma: no branch
        # let `builds` validate the new defaults!
        fn(fn, **_new_defaults)

    _zen_dataclass: Optional[DataclassOptions] = _new_defaults.pop("zen_dataclass")
    if _zen_dataclass is None:
        _zen_dataclass = {}

    if _frozen is True:
        warnings.warn(
            HydraZenDeprecationWarning(
                "Specifying `builds(..., frozen=<...>)` is deprecated. Instead, "
                "specify `builds(..., zen_dataclass={'frozen': <...>})"
            ),
            stacklevel=2,
        )

        _zen_dataclass["frozen"] = _frozen

    _zen_dataclass = parse_dataclass_options(_zen_dataclass)

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        merged_kwargs: dict[str, Any] = {}
        _dataclass: Optional[DataclassOptions] = kwargs.pop("zen_dataclass", None)

        if _dataclass is None:
            _new_defaults["zen_dataclass"] = _zen_dataclass
        else:
            _new_defaults["zen_dataclass"] = {**_zen_dataclass, **_dataclass}

        merged_kwargs.update(_new_defaults)
        merged_kwargs.update(kwargs)
        return cast(Any, fn(*args, **merged_kwargs))

    return cast(
        Union[
            FullBuilds[T],
            PBuilds[T],
            StdBuilds[T],
        ],
        wrapped,
    )
