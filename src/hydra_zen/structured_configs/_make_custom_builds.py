# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
import inspect
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional, Union, cast, overload

from typing_extensions import Final, Literal

from hydra_zen.typing import ZenWrappers
from hydra_zen.typing._builds_overloads import FullBuilds, PBuilds, StdBuilds
from hydra_zen.typing._implementations import ZenConvert

from ._implementations import builds

__all__ = ["make_custom_builds_fn"]


_builds_sig = inspect.signature(builds)
__BUILDS_DEFAULTS: Final[Dict[str, Any]] = {
    name: p.default
    for name, p in _builds_sig.parameters.items()
    if p.kind is p.KEYWORD_ONLY
}
del _builds_sig


# partial=False, pop-sig=True
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    populate_full_signature: Literal[True],
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> FullBuilds:  # pragma: no cover
    ...


# partial=True, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[True],
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    populate_full_signature: bool = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> PBuilds:  # pragma: no cover
    ...


# partial=False, pop-sig=False
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> StdBuilds:  # pragma: no cover
    ...


# partial=False, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Literal[False, None] = ...,
    populate_full_signature: bool,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> Union[FullBuilds, StdBuilds]:  # pragma: no cover
    ...


# partial=bool, pop-sig=False
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Union[bool, None],
    populate_full_signature: Literal[False] = ...,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> Union[PBuilds, StdBuilds]:  # pragma: no cover
    ...


# partial=bool, pop-sig=bool
@overload
def make_custom_builds_fn(
    *,
    zen_partial: Union[bool, None],
    populate_full_signature: bool,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = ...,
    zen_meta: Optional[Mapping[str, Any]] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
    frozen: bool = ...,
    zen_convert: Optional[ZenConvert] = ...,
) -> Union[FullBuilds, PBuilds, StdBuilds]:  # pragma: no cover
    ...


def make_custom_builds_fn(
    *,
    zen_partial: Optional[bool] = None,
    populate_full_signature: bool = False,
    zen_wrappers: ZenWrappers[Callable[..., Any]] = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    zen_convert: Optional[ZenConvert] = None,
) -> Union[FullBuilds, PBuilds, StdBuilds]:
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

    zen_convert : Optional[ZenConvert]
        A dictionary that modifies hydra-zen's value and type conversion behavior.
        Consists of the following optional key-value pairs (:ref:`zen-convert`):

        - `dataclass` : `bool` (default=True):
            If `True` any dataclass type/instance without a
            `_target_` field is automatically converted to a targeted config
            that will instantiate to that type/instance. Otherwise the dataclass
            type/instance will be passed through as-is.

    hydra_recursive : Optional[bool], optional (default=True)
        Specifies a new the default value for ``builds(..., hydra_recursive=<..>)``

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Specifies a new the default value for ``builds(..., hydra_convert=<..>)``

    frozen : bool, optional (default=False)
        Specifies a new the default value for ``builds(..., frozen=<..>)``

    Returns
    -------
    custom_builds
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

    >>> instantiate(conf, x="c")  # violates annotation: Literal["a", "b"]
    <Validation error: "c" is not "a" or "b">
    """

    excluded_fields = frozenset({"dataclass_name", "hydra_defaults", "builds_bases"})
    LOCALS = locals()

    # Ensures that new defaults added to `builds` must be reflected
    # in the signature of `make_custom_builds_fn`.
    assert (set(__BUILDS_DEFAULTS) - excluded_fields) <= set(LOCALS)

    _new_defaults = {
        name: LOCALS[name] for name in __BUILDS_DEFAULTS if name not in excluded_fields
    }

    # let `builds` validate the new defaults!
    builds(builds, **_new_defaults)

    @wraps(builds)
    def wrapped(*args: Any, **kwargs: Any):
        merged_kwargs: Dict[str, Any] = {}
        merged_kwargs.update(_new_defaults)
        merged_kwargs.update(kwargs)
        return builds(*args, **merged_kwargs)

    setattr(wrapped, "_zen_convert", zen_convert)
    return cast(Union[FullBuilds, PBuilds, StdBuilds], wrapped)
