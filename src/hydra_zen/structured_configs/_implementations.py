# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
import inspect
import sys
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import (
    MISSING,
    Field,
    InitVar,
    dataclass,
    field,
    fields,
    is_dataclass,
    make_dataclass,
)
from enum import Enum
from functools import wraps
from itertools import chain
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

from omegaconf import DictConfig, ListConfig
from typing_extensions import Final, Literal, TypeGuard

from hydra_zen._compatibility import (
    HYDRA_SUPPORTED_PRIMITIVES,
    HYDRA_SUPPORTS_PARTIAL,
    PATCH_OMEGACONF_830,
    ZEN_SUPPORTED_PRIMITIVES,
)
from hydra_zen.errors import (
    HydraZenDeprecationWarning,
    HydraZenUnsupportedPrimitiveError,
    HydraZenValidationError,
)
from hydra_zen.funcs import get_obj, partial, zen_processing
from hydra_zen.structured_configs import _utils
from hydra_zen.typing import (
    Builds,
    Importable,
    Just,
    Partial,
    PartialBuilds,
    SupportedPrimitive,
)
from hydra_zen.typing._implementations import (
    DataClass,
    HasPartialTarget,
    HasTarget,
    _DataClass,
)

from ._value_conversion import ZEN_VALUE_CONVERSION

_T = TypeVar("_T")
_T2 = TypeVar("_T2", bound=Callable)
ZenWrapper = Union[
    None,
    Builds[Callable[[_T2], _T2]],
    PartialBuilds[Callable[[_T2], _T2]],
    Just[Callable[[_T2], _T2]],
    Type[Builds[Callable[[_T2], _T2]]],
    Type[PartialBuilds[Callable[[_T2], _T2]]],
    Type[Just[Callable[[_T2], _T2]]],
    Callable[[_T2], _T2],
    str,
]
if TYPE_CHECKING:  # pragma: no cover
    ZenWrappers = Union[ZenWrapper, Sequence[ZenWrapper]]
else:
    ZenWrappers = TypeVar("ZenWrappers")

# Hydra-specific fields
_TARGET_FIELD_NAME: Final[str] = "_target_"
_PARTIAL_FIELD_NAME: Final[str] = "_partial_"
_RECURSIVE_FIELD_NAME: Final[str] = "_recursive_"
_CONVERT_FIELD_NAME: Final[str] = "_convert_"
_POS_ARG_FIELD_NAME: Final[str] = "_args_"

_HYDRA_FIELD_NAMES: FrozenSet[str] = frozenset(
    (
        _TARGET_FIELD_NAME,
        _RECURSIVE_FIELD_NAME,
        _CONVERT_FIELD_NAME,
        _POS_ARG_FIELD_NAME,
    )
)

# hydra-zen-specific fields
_ZEN_PROCESSING_LOCATION: Final[str] = _utils.get_obj_path(zen_processing)
_ZEN_TARGET_FIELD_NAME: Final[str] = "_zen_target"
_ZEN_PARTIAL_TARGET_FIELD_NAME: Final[str] = "_zen_partial"
_META_FIELD_NAME: Final[str] = "_zen_exclude"
_ZEN_WRAPPERS_FIELD_NAME: Final[str] = "_zen_wrappers"
_JUST_FIELD_NAME: Final[str] = "path"
# TODO: add _JUST_Target

# signature param-types
_POSITIONAL_ONLY: Final = inspect.Parameter.POSITIONAL_ONLY
_POSITIONAL_OR_KEYWORD: Final = inspect.Parameter.POSITIONAL_OR_KEYWORD
_VAR_POSITIONAL: Final = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY: Final = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD: Final = inspect.Parameter.VAR_KEYWORD

_builtin_function_or_method_type = type(len)


def _get_target(x):
    return getattr(x, _TARGET_FIELD_NAME)


def _target_as_kwarg_deprecation(func: _T2) -> Callable[..., _T2]:
    @wraps(func)
    def wrapped(*args, **kwargs):
        if not args and "target" in kwargs:
            # builds(target=<>, ...) is deprecated
            warnings.warn(
                HydraZenDeprecationWarning(
                    "Specifying the target of `builds` as a keyword argument is deprecated "
                    "as of 2021-10-27. Change `builds(target=<target>, ...)` to `builds(<target>, ...)`."
                    "\n\nThis will be an error in hydra-zen 1.0.0, or by 2021-01-27 — whichever "
                    "comes first.\n\nNote: This deprecation does not impact yaml configs "
                    "produced by `builds`."
                ),
                stacklevel=2,
            )
            target = kwargs.pop("target")
            return func(target, *args, **kwargs)
        return func(*args, **kwargs)

    return wrapped


def _hydra_partial_deprecation(func: _T2) -> Callable[..., _T2]:
    @wraps(func)
    def wrapped(*args, **kwargs):
        if "hydra_partial" in kwargs:
            if "zen_partial" in kwargs:
                raise TypeError(
                    "Both `hydra_partial` and `zen_partial` are specified. "
                    "Specifying `hydra_partial` is deprecated, use `zen_partial` "
                    "instead."
                )

            # builds(..., hydra_partial=...) is deprecated
            warnings.warn(
                HydraZenDeprecationWarning(
                    "The argument `hydra_partial` is deprecated as of 2021-10-27.\n"
                    "Change `builds(..., hydra_partial=<..>)` to `builds(..., zen_partial=<..>)`."
                    "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-27 — whichever "
                    "comes first.\n\nNote: This deprecation does not impact yaml configs "
                    "produced by `builds`."
                ),
                stacklevel=2,
            )
            kwargs["zen_partial"] = kwargs.pop("hydra_partial")
        return func(*args, **kwargs)

    return wrapped


def mutable_value(x: _T) -> _T:
    """Used to set a mutable object as a default value for a field
    in a dataclass.

    This is an alias for ``field(default_factory=lambda: type(x)(x))``

    Note that ``type(x)(...)`` serves to make a copy

    Examples
    --------
    >>> from hydra_zen import mutable_value
    >>> from dataclasses import dataclass

    See https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    >>> @dataclass  # doctest: +SKIP
    ... class HasMutableDefault:
    ...     a_list: list  = [1, 2, 3]  # error: mutable default

    Using `mutable_value` to specify the default list:

    >>> @dataclass
    ... class HasMutableDefault:
    ...     a_list: list  = mutable_value([1, 2, 3])  # ok

    >>> x = HasMutableDefault()
    >>> x.a_list.append(-1)  # does not append to `HasMutableDefault.a_list`
    >>> x
    HasMutableDefault(a_list=[1, 2, 3, -1])
    >>> HasMutableDefault()
    HasMutableDefault(a_list=[1, 2, 3])"""
    cast = type(x)  # ensure that we return a copy of the default value
    x = sanitize_collection(x)
    return field(default_factory=lambda: cast(x))


Field_Entry = Tuple[str, type, Field]


# Alternate form, from PEP proposal:
# https://github.com/microsoft/pyright/blob/master/specs/dataclass_transforms.md
#
# This enables static checkers to work with third-party decorators that create
# dataclass-like objects
def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
    # If used within a stub file, the following implementation can be
    # replaced with "...".
    return lambda a: a


@__dataclass_transform__()
def hydrated_dataclass(
    target: Callable,
    *pos_args: SupportedPrimitive,
    zen_partial: bool = False,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    **_kw,  # reserved to deprecate hydra_partial
) -> Callable[[Type[_T]], Type[_T]]:
    """A decorator that uses `builds` to create a dataclass with the appropriate
    Hydra-specific fields for specifying a targeted config [1]_.

    This provides similar functionality to `builds`, but enables a user to define
    a config explicitly using the :func:`dataclasses.dataclass` syntax, which can
    enable enhanced static analysis of the resulting config.

    Parameters
    ----------
    hydra_target : T (Callable)
        The target-object to be configured. This is a required, positional-only argument.

    *pos_args : SupportedPrimitive
        Positional arguments passed as ``hydra_target(*pos_args, ...)`` upon instantiation.

        Arguments specified positionally are not included in the dataclass' signature and
        are stored as a tuple bound to in the ``_args_`` field.

    zen_partial : bool, optional (default=False)
        If ``True``, then the resulting config will instantiate as
        ``functools.partial(hydra_target, *pos_args, **kwargs_for_target)`` rather than
        ``hydra_target(*pos_args, **kwargs_for_target)``. Thus this enables the
        partial-configuration of objects.

        Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together
        will populate the config's signature only with parameters that: are explicitly
        specified by the user, or that have default values specified in the target's
        signature. I.e. it is presumed that un-specified parameters that have no
        default values are to be excluded from the config.

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(hydra_target)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [2]_ or targeted
        configs.

    zen_meta : Optional[Mapping[str, SupportedPrimitive]]
        Specifies field-names and corresponding values that will be included in the
        resulting config, but that will *not* be used to builds ``<hydra_target>``
        via instantiation. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting config's signature and fields will be populated
        according to the signature of ``hydra_target``; values also specified in
        ``**kwargs_for_target`` take precedent.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this config [3]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
          a trace of OmegaConf containers.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    frozen : bool, optional (default=False)
        If ``True``, the resulting config will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise
        :py:class:`dataclasses.FrozenInstanceError` at runtime.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    Unlike `builds`, `hydrated_dataclass` enables config fields to be set explicitly
    with custom type annotations. Additionally, the resulting config' attributes
    can be analyzed by static tooling, which can help to warn about errors prior
    to running one's code.

    For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [4] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import hydrated_dataclass, instantiate

    Here, we specify a config that is designed to "build" a dictionary
    upon instantiation

    >>> @hydrated_dataclass(target=dict)
    ... class DictConf:
    ...     x: int = 2
    ...     y: str = 'hello'

    >>> instantiate(DictConf(x=10))  # override default `x`
    {'x': 10, 'y': 'hello'}

    For more detailed examples, refer to `builds`.
    """

    if "hydra_partial" in _kw:
        if zen_partial is True:
            raise TypeError(
                "Both `hydra_partial` and `zen_partial` are specified. "
                "Specifying `hydra_partial` is deprecated, use `zen_partial` "
                "instead."
            )

        # builds(..., hydra_partial=...) is deprecated
        warnings.warn(
            HydraZenDeprecationWarning(
                "The argument `hydra_partial` is deprecated as of 2021-10-27.\n"
                "Change `builds(..., hydra_partial=<..>)` to `builds(..., zen_partial=<..>)`."
                "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-27 — whichever "
                "comes first.\n\nNote: This deprecation does not impact yaml configs "
                "produced by `builds`."
            ),
            stacklevel=2,
        )
        zen_partial = _kw.pop("hydra_partial")
    if _kw:
        raise TypeError(
            f"hydrated_dataclass got an unexpected argument: {', '.join(_kw)}"
        )

    def wrapper(decorated_obj: Any) -> Any:

        if not isinstance(decorated_obj, type):
            raise NotImplementedError(
                "Class instances are not supported by `hydrated_dataclass`."
            )

        # TODO: We should mutate `decorated_obj` directly like @dataclass does.
        #       Presently, we create an intermediate dataclass that we inherit
        #       from, which gets the job done for the most part but there are
        #       practical differences. E.g. you cannot delete an attribute that
        #       was declared in the definition of `decorated_obj`.
        decorated_obj = cast(Any, decorated_obj)
        decorated_obj = dataclass(frozen=frozen)(decorated_obj)

        if PATCH_OMEGACONF_830 and 2 < len(decorated_obj.__mro__):
            parents = decorated_obj.__mro__[1:-1]
            # this class inherits from a parent
            for field_ in fields(decorated_obj):
                if field_.default_factory is not MISSING and any(
                    hasattr(p, field_.name) for p in parents
                ):
                    # TODO: update error message with fixed omegaconf version
                    _value = field_.default_factory()
                    raise HydraZenValidationError(
                        "This config will not instantiate properly.\nThis is due to a "
                        "known bug in omegaconf: The config specifies a "
                        f"default-factory for field {field_.name}, and inherits from a "
                        "parent that specifies the same field with a non-factory value "
                        "-- the parent's value will take precedence.\nTo circumvent "
                        f"this, specify {field_.name} using: "
                        f"`builds({type(_value).__name__}, {_value})`\n\nFor more "
                        "information, see: https://github.com/omry/omegaconf/issues/830"
                    )

        if populate_full_signature:
            # we need to ensure that the fields specified via the class definition
            # take precedence over the fields that will be auto-populated by builds
            kwargs = {
                f.name: f.default if f.default is not MISSING else f.default_factory()
                for f in fields(decorated_obj)
                if not (f.default is MISSING and f.default_factory is MISSING)
                and f.name not in _HYDRA_FIELD_NAMES
                and not f.name.startswith("_zen_")
            }
        else:
            kwargs = {}

        return builds(
            target,
            *pos_args,
            **kwargs,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            zen_wrappers=zen_wrappers,
            zen_partial=zen_partial,
            zen_meta=zen_meta,
            builds_bases=(decorated_obj,),
            dataclass_name=decorated_obj.__name__,
            frozen=frozen,
        )

    return wrapper


def just(obj: Importable) -> Type[Just[Importable]]:
    """Produces a config that, when instantiated by Hydra, "just" returns the un-instantiated target-object.

    Parameters
    ----------
    obj : Importable
        The object that will be instantiated from this config.

    Returns
    -------
    config : Type[Just[Importable]]

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    make_config: Creates a general config with customized field names, default values, and annotations.

    Notes
    -----
    The configs produced by `just` introduce an explicit dependency on hydra-zen. I.e.
    hydra-zen must be installed in order to instantiate any config that used `just`.

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import just, instantiate, to_yaml

    >>> Conf = just(range)
    >>> instantiate(Conf) is range
    True

    The config produced by `just` describes how to import the target,
    not how to instantiate/call the object.

    >>> print(to_yaml(Conf))
    _target_: hydra_zen.funcs.get_obj
    path: builtins.range

    **Auto-Application of just**

    Both `builds` and `make_config` will automatically apply `just` to default values
    that are function-objects or class objects. E.g. in the following example `just`
    will be applied to ``sum``.

    >>> from hydra_zen import make_config
    >>> Conf2 = make_config(data=[1, 2, 3], reduction_fn=sum)

    >>> print(to_yaml(Conf2))
    data:
    - 1
    - 2
    - 3
    reduction_fn:
      _target_: hydra_zen.funcs.get_obj
      path: builtins.sum

    >>> conf = instantiate(Conf2)
    >>> conf.reduction_fn(conf.data)
    6
    """
    try:
        obj_path = _utils.get_obj_path(obj)
    except AttributeError:
        raise AttributeError(
            f"`just({obj})`: `obj` is not importable; it is missing the attributes `__module__` and/or `__qualname__`"
        )

    out_class = make_dataclass(
        ("Just_" + _utils.safe_name(obj)),
        [
            (
                _TARGET_FIELD_NAME,
                str,
                _utils.field(default=_utils.get_obj_path(get_obj), init=False),
            ),
            (
                "path",
                str,
                _utils.field(
                    default=obj_path,
                    init=False,
                ),
            ),
        ],
    )
    out_class.__doc__ = (
        f"A structured config designed to return {obj} when it is instantiated by hydra"
    )

    return cast(Type[Just[Importable]], out_class)


_KEY_ERROR_PREFIX = "Configuring dictionary key:"


def _is_ufunc(value) -> bool:
    # checks without importing numpy
    numpy = sys.modules.get("numpy")
    if numpy is None:  # pragma: no cover
        # we do actually cover this branch some runs of our CI,
        # but our coverage job installs numpy
        return False
    return isinstance(value, numpy.ufunc)  # type: ignore


def sanitized_default_value(
    value: Any,
    allow_zen_conversion: bool = True,
    *,
    error_prefix: str = "",
    field_name: str = "",
    structured_conf_permitted: bool = True,
) -> Any:
    value = sanitize_collection(value)

    if (
        structured_conf_permitted
        and callable(value)
        and (
            inspect.isfunction(value)
            or (not is_dataclass(value) and inspect.isclass(value))
            or isinstance(value, _builtin_function_or_method_type)
            or _is_ufunc(value)
        )
    ):
        return just(value)
    resolved_value = value
    type_of_value = type(resolved_value)

    # we don't use isinstance because we don't permit subclasses of supported
    # primitives
    if allow_zen_conversion and type_of_value in ZEN_SUPPORTED_PRIMITIVES:
        type_ = type(resolved_value)
        conversion_fn = ZEN_VALUE_CONVERSION.get(type_)

        if conversion_fn is not None:
            resolved_value = conversion_fn(resolved_value)
            type_of_value = type(resolved_value)

    if type_of_value in HYDRA_SUPPORTED_PRIMITIVES or (
        structured_conf_permitted
        and (
            is_dataclass(resolved_value)
            or isinstance(resolved_value, (Enum, ListConfig, DictConfig))
        )
    ):
        return resolved_value

    if field_name:
        field_name = f", for field `{field_name}`,"

    err_msg = (
        error_prefix
        + f" The configured value {value}{field_name} is not supported by Hydra -- "
        f"serializing or instantiating this config would ultimately result in an error."
    )

    if structured_conf_permitted:
        err_msg += f"\n\nConsider using `hydra_zen.builds({type(value)}, ...)` to "
        "create a config for this particular value."

    raise HydraZenUnsupportedPrimitiveError(err_msg)


def sanitize_collection(x: _T) -> _T:
    """Pass contents of lists, tuples, or dicts through sanitized_default_values"""
    type_x = type(x)
    if type_x in {list, tuple}:
        return type_x(sanitized_default_value(_x) for _x in x)  # type: ignore
    elif type_x is dict:
        return {
            # Hydra doesn't permit structured configs for keys, thus we only
            # support its basic primitives here.
            sanitized_default_value(
                k,
                allow_zen_conversion=False,
                structured_conf_permitted=False,
                error_prefix=_KEY_ERROR_PREFIX,
            ): sanitized_default_value(v)
            for k, v in x.items()  # type: ignore
        }
    else:
        # pass-through
        return x


def sanitized_field(
    value: Any,
    init=True,
    allow_zen_conversion: bool = True,
    *,
    error_prefix: str = "",
    field_name: str = "",
    _mutable_default_permitted: bool = True,
) -> Field:
    type_value = type(value)
    if (
        type_value in _utils.KNOWN_MUTABLE_TYPES
        and type_value in HYDRA_SUPPORTED_PRIMITIVES
    ):
        if _mutable_default_permitted:
            return cast(Field, mutable_value(value))

        value = builds(type(value), value)

    return _utils.field(
        default=sanitized_default_value(
            value,
            allow_zen_conversion=allow_zen_conversion,
            error_prefix=error_prefix,
            field_name=field_name,
        ),
        init=init,
    )


# overloads when `zen_partial=False`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[False] = False,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Type[_DataClass], ...] = (),
    frozen: bool = False,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


# overloads when `zen_partial=True`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: Literal[True],
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Type[_DataClass], ...] = (),
    frozen: bool = False,
    **kwargs_for_target: SupportedPrimitive,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


# overloads when `zen_partial: bool`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: SupportedPrimitive,
    zen_partial: bool,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Type[_DataClass], ...] = (),
    frozen: bool = False,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[
    Type[Builds[Importable]], Type[PartialBuilds[Importable]]
]:  # pragma: no cover
    ...


@_hydra_partial_deprecation
@_target_as_kwarg_deprecation
def builds(
    *pos_args: Any,
    zen_partial: bool = False,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, SupportedPrimitive]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    builds_bases: Tuple[Type[_DataClass], ...] = (),
    dataclass_name: Optional[str] = None,
    **kwargs_for_target: SupportedPrimitive,
) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
    """builds(hydra_target, /, *pos_args, zen_partial=False, zen_meta=None,
    hydra_recursive=None, populate_full_signature=False, hydra_convert=None,
    frozen=False, dataclass_name=None, builds_bases=(), **kwargs_for_target)

    Returns a config, which describes how to instantiate/call ``<hydra_target>`` with
    both user-specified and auto-populated parameter values.

    Consult the Examples section of the docstring to see the various features of
    `builds` in action.

    Parameters
    ----------
    hydra_target : T (Callable)
        The target object to be configured. This is a required, **positional-only**
        argument.

    *pos_args : SupportedPrimitive
        Positional arguments passed as ``<hydra_target>(*pos_args, ...)`` upon
        instantiation.

        Arguments specified positionally are not included in the dataclass' signature
        and are stored as a tuple bound to in the ``_args_`` field.

    **kwargs_for_target : SupportedPrimitive
        The keyword arguments passed as ``<hydra_target>(..., **kwargs_for_target)``
        upon instantiation.

        The arguments specified here solely determine the signature of the resulting
        config, unless ``populate_full_signature=True`` is specified (see below).

        Named parameters of the forms that have the prefixes ``hydra_``, ``zen_`` or
        ``_zen_`` are reserved to ensure future-compatibility, and thus cannot be
        specified by the user.

    zen_partial : bool, optional (default=False)
        If ``True``, then the resulting config will instantiate as
        ``functools.partial(<hydra_target>, *pos_args, **kwargs_for_target)``. Thus
        this enables the partial-configuration of objects.

        Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together
        will populate the config's signature only with parameters that: are explicitly
        specified by the user, or that have default values specified in the target's
        signature. I.e. it is presumed that un-specified parameters that have no
        default values are to be excluded from the config.

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(<hydra_target>)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [2]_ or targeted
        configs.

    zen_meta : Optional[Mapping[str, SupportedPrimitive]]
        Specifies field-names and corresponding values that will be included in the
        resulting config, but that will *not* be used to instantiate
        ``<hydra_target>``. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting config's signature and fields will be populated
        according to the signature of ``<hydra_target>``; values also specified in
        ``**kwargs_for_target`` take precedent.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this config [3]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive, omegaconf-specific objects passed to
        ``<hydra_target>`` [4]_.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
          a trace of OmegaConf containers.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    frozen : bool, optional (default=False)
        If ``True``, the resulting config will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise
        :py:class:`dataclasses.FrozenInstanceError` at runtime.

    builds_bases : Tuple[DataClass, ...]
        Specifies a tuple of parent classes that the resulting config inherits from.
        A ``PartialBuilds`` class (resulting from ``zen_partial=True``) cannot be a
        parent of a ``Builds`` class (i.e. where `zen_partial=False` was specified).

    dataclass_name : Optional[str]
        If specified, determines the name of the returned class object.

    Returns
    -------
    Config : Type[Builds[Type[T]]] | Type[PartialBuilds[Type[T]]]
        A structured config that describes how to build ``hydra_target``.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    The resulting "config" is a dataclass-object [5]_ with Hydra-specific attributes
    attached to it [1]_.

    Using any of the ``zen_xx`` features will result in a config that depends
    explicitly on hydra-zen. I.e. hydra-zen must be installed in order to
    instantiate the resulting config, including its yaml version.

    For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

    Type annotations are inferred from the target's signature and are only
    retained if they are compatible with Hydra's limited set of supported
    annotations; otherwise an annotation is automatically 'broadened' until
    it is made compatible with Hydra.

    `builds` provides runtime validation of user-specified arguments against
    the target's signature. E.g. specifying mis-named arguments or too many
    arguments will cause `builds` to raise.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [4] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [5] https://docs.python.org/3/library/dataclasses.html
    .. [6] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation

    See Also
    --------
    instantiate: Instantiates a configuration created by `builds`, returning the instantiated target.
    make_custom_builds_fn: Returns the `builds` function, but one with customized default values.
    make_config: Creates a general config with customized field names, default values, and annotations.
    get_target: Returns the target-object from a targeted structured config.
    just: Produces a config that, when instantiated by Hydra, "just" returns the un-instantiated target-object.
    to_yaml: Serialize a config as a yaml-formatted string.

    Examples
    --------
    **Basic Usage**

    Lets create a basic config that describes how to 'build' a particular dictionary.

    >>> from hydra_zen import builds, instantiate
    >>> Conf = builds(dict, a=1, b='x')

    The resulting config is a dataclass with the following signature and attributes:

    >>> Conf  # signature: Conf(a: Any = 1, b: Any = 'x')
    <class 'types.Builds_dict'>
    >>> Conf.a
    1
    >>> Conf.b
    'x'

    The `instantiate` function is used to enact this build – to create the dictionary.

    >>> instantiate(Conf)  # calls: `dict(a=1, b='x')`
    {'a': 1, 'b': 'x'}

    The default parameters that we provided can be overridden.

    >>> new_conf = Conf(a=10, b="hi")  # an instance of our dataclass
    >>> instantiate(new_conf)  # calls: `dict(a=10, b='hi')`
    {'a': 10, 'b': 'hi'}

    Positional arguments can be provided too.

    >>> Conf = builds(len, [1, 2, 3])  # specifying positional arguments
    >>> instantiate(Conf)
    3

    **Creating a Partial Config**

    `builds` can be used to partially-configure a target. Let's
    create a config for the following function

    >>> def a_two_tuple(x: int, y: float): return x, y

    such that we only configure the parameter ``x``.

    >>> PartialConf = builds(a_two_tuple, x=1, zen_partial=True)  # configures only `x`

    Instantiating this conf will return ``functools.partial(a_two_tuple, x=1)``.

    >>> partial_func = instantiate(PartialConf)
    >>> partial_func
    functools.partial(<function a_two_tuple at 0x00000220A7820EE0>, x=1)

    And thus the remaining parameter can be provided post-instantiation.

    >>> partial_func(y=22.0)  # providing the remaining parameter
    (1, 22.0)

    **Auto-populating parameters**

    The configurable parameters of a target can be auto-populated in our config.
    Suppose we want to configure the following function.

    >>> def f(x: bool, y: str = 'foo'): return x, y

    The following config will have a signature that matches ``f``; the
    annotations and default values of the parameters of ``f`` are explicitly
    incorporated into the config.

    >>> Conf = builds(f, populate_full_signature=True)  # signature: `Builds_f(x: bool, y: str = 'foo')`
    >>> Conf.y
    'foo'

    Annotations will be used by Hydra to provide limited runtime type-checking during
    instantiation. Here, we'll pass a float for ``x``, which expects a boolean value.

    >>> instantiate(Conf(x=10.0))
    ValidationError: Value '10.0' is not a valid bool (type float)
        full_key: x
        object_type=Builds_f

    **Composing configs via inheritance**

    Because a config produced via `builds` is simply a class-object, we can
    compose configs via class inheritance.

    >>> ParentConf = builds(dict, a=1, b=2)
    >>> ChildConf = builds(dict, b=-2, c=-3, builds_bases=(ParentConf,))
    >>> instantiate(ChildConf)
    {'a': 1, 'b': -2, 'c': -3}
    >>> issubclass(ChildConf, ParentConf)
    True

    .. _builds-validation:

    **Runtime validation perfomed by builds**

    Misspelled parameter names and other invalid configurations for the target’s
    signature will be caught by `builds`, so that such errors are caught prior to
    instantiation.

    >>> def func(a_number: int): pass

    >>> builds(func, a_nmbr=2)  # misspelled parameter name
    TypeError: Building: func ..

    >>> builds(func, 1, 2)  # too many arguments
    TypeError: Building: func ..

    >>> BaseConf = builds(func, a_number=2)
    >>> builds(func, 1, builds_bases=(BaseConf,))  # too many args (via inheritance)
    TypeError: Building: func ..

    .. _meta-field:

    **Using meta-fields**

    Meta-fields are fields that are included in a config but are excluded by the
    instantiation process. Thus arbitrary metadata can be attached to a config.

    Let's create a config whose fields reference a meta-field via
    relative-interpolation [6]_.

    >>> Conf = builds(dict, a="${.s}", b="${.s}", zen_meta=dict(s=-10))
    >>> instantiate(Conf)
    {'a': -10, 'b': -10}
    >>> instantiate(Conf, s=2)
    {'a': 2, 'b': 2}

    .. _zen-wrapper:

    **Using zen-wrappers**

    Zen-wrappers enables us to make arbitrary changes to ``<hydra_target>``, its inputs,
    and/or its outputs during the instantiation process.

    Let's use a wrapper to add a unit-conversion step to a config. We'll modify a
    config that builds a function, which converts a temperature in Farenheit to
    Celcius, and add a wrapper it so that it will convert from Farenheit to Kelvin
    instead.

    >>> def faren_to_celsius(temp_f):  # our target
    ...     return ((temp_f - 32) * 5) / 9

    >>> def change_celcius_to_kelvin(celc_func):  # our wrapper
    ...     def wraps(*args, **kwargs):
    ...         return 273.15 + celc_func(*args, **kwargs)
    ...     return wraps

    >>> AsCelcius = builds(faren_to_celsius)
    >>> AsKelvin = builds(faren_to_celsius, zen_wrappers=change_celcius_to_kelvin)
    >>> instantiate(AsCelcius, temp_f=32)
    0.0
    >>> instantiate(AsKelvin, temp_f=32)
    273.15

    **Creating a frozen config**

    Let's create a config object whose instances will by "frozen" (i.e., immutable).

    >>> RouterConfig = builds(dict, ip_address=None, frozen=True)
    >>> my_router = RouterConfig(ip_address="192.168.56.1")  # an immutable instance

    Attempting to overwrite the attributes of ``my_router`` will raise.

    >>> my_router.ip_address = "148.109.37.2"
    FrozenInstanceError: cannot assign to field 'ip_address'

    **Support for partial'd objects**

    Specifying ``builds(functools.partial(<target>, ...), ...)`` is supported; `builds`
    will automatically "unpack" a partial'd object that is passed as its target.

    >>> import functools
    >>> partiald_dict = functools.partial(dict, a=1, b=2)
    >>> Conf = builds(partiald_dict)  # signature: (a = 1, b = 2)
    >>> Conf.a, Conf.b
    (1, 2)
    >>> instantiate(Conf(a=-4))  # equivalent to calling: `partiald_dict(a=-4)`
    {'a': -4, 'b': 2}
    """

    if not pos_args and not kwargs_for_target:
        # `builds()`
        raise TypeError(
            "builds() missing 1 required positional argument: 'hydra_target'"
        )
    elif not pos_args:
        # `builds(hydra_target=int)`
        raise TypeError(
            "builds() missing 1 required positional-only argument: 'hydra_target'"
            "\nChange `builds(hydra_target=<target>, ...)` to `builds(<target>, ...)`"
        )

    target, *_pos_args = pos_args

    if isinstance(target, functools.partial):
        # partial'd args must come first, then user-specified args
        # otherwise, the parial'd args will take precedent, which
        # does not align with the behavior of partial itself
        _pos_args = list(target.args) + _pos_args
        kwargs_for_target = {**target.keywords, **kwargs_for_target}
        target = target.func

    BUILDS_ERROR_PREFIX = _utils.building_error_prefix(target)

    del pos_args

    if not callable(target):
        raise TypeError(
            BUILDS_ERROR_PREFIX
            + "In `builds(<target>, ...), `<target>` must be callable/instantiable"
        )

    if not isinstance(populate_full_signature, bool):
        raise TypeError(
            f"`populate_full_signature` must be a boolean type, got: {populate_full_signature}"
        )

    if hydra_recursive is not None and not isinstance(hydra_recursive, bool):
        raise TypeError(
            f"`hydra_recursive` must be a boolean type, got {hydra_recursive}"
        )

    if not isinstance(zen_partial, bool):
        raise TypeError(f"`zen_partial` must be a boolean type, got: {zen_partial}")

    if hydra_convert is not None and hydra_convert not in {"none", "partial", "all"}:
        raise ValueError(
            f"`hydra_convert` must be 'none', 'partial', or 'all', got: {hydra_convert}"
        )

    if not isinstance(frozen, bool):
        raise TypeError(f"frozen must be a bool, got: {frozen}")

    if dataclass_name is not None and not isinstance(dataclass_name, str):
        raise TypeError(
            f"`dataclass_name` must be a string or None, got: {dataclass_name}"
        )

    if any(not (is_dataclass(_b) and isinstance(_b, type)) for _b in builds_bases):
        raise TypeError("All `build_bases` must be a tuple of dataclass types")

    if zen_meta is None:
        zen_meta = {}

    if not isinstance(zen_meta, Mapping):
        raise TypeError(
            f"`zen_meta` must be a mapping (e.g. a dictionary), got: {zen_meta}"
        )

    if any(not isinstance(_key, str) for _key in zen_meta):
        raise TypeError(
            f"`zen_meta` must be a mapping whose keys are strings, got key(s):"
            f" {','.join(str(_key) for _key in zen_meta if not isinstance(_key, str))}"
        )

    if zen_wrappers is not None:
        if not isinstance(zen_wrappers, Sequence) or isinstance(zen_wrappers, str):
            zen_wrappers = (zen_wrappers,)

        validated_wrappers: Sequence[Union[str, Builds]] = []
        for wrapper in zen_wrappers:
            if wrapper is None:
                continue
            # We are intentionally keeping each condition branched
            # so that test-coverage will be checked for each one
            if is_builds(wrapper):
                # If Hydra's locate function starts supporting importing literals
                # – or if we decide to ship our own locate function –
                # then we should get the target of `wrapper` and make sure it is callable
                if is_just(wrapper):
                    # `zen_wrappers` handles importing string; we can
                    # elimintate the indirection of Just and "flatten" this
                    # config
                    validated_wrappers.append(getattr(wrapper, _JUST_FIELD_NAME))
                else:
                    if hydra_recursive is False:
                        warnings.warn(
                            "A structured config was supplied for `zen_wrappers`. Its parent config has "
                            "`hydra_recursive=False`.\n If this value is not toggled to `True`, the config's "
                            "instantiation will result in an error"
                        )
                    validated_wrappers.append(wrapper)

            elif callable(wrapper):
                validated_wrappers.append(_utils.get_obj_path(wrapper))

            elif isinstance(wrapper, str):
                # Assumed that wrapper is either a valid omegaconf-style interpolation string
                # or a "valid" path for importing an object. The latter seems hopeless for validating:
                # https://stackoverflow.com/a/47538106/6592114
                # so we can't make any assurances here.
                validated_wrappers.append(wrapper)
            else:
                raise TypeError(
                    f"`zen_wrappers` requires a callable, targeted config, or a string, got: {wrapper}"
                )

        del zen_wrappers
        validated_wrappers = tuple(validated_wrappers)
    else:
        validated_wrappers = ()

    # Check for reserved names
    for _name in chain(kwargs_for_target, zen_meta):
        if _name in _HYDRA_FIELD_NAMES:
            err_msg = f"The field-name specified via `builds(..., {_name}=<...>)` is reserved by Hydra."
            if _name != _TARGET_FIELD_NAME:
                raise ValueError(
                    err_msg
                    + f" You can set this parameter via `builds(..., hydra_{_name[1:-1]}=<...>)`"
                )
            else:
                raise ValueError(err_msg)
        if _name.startswith(("hydra_", "_zen_", "zen_")):
            raise ValueError(
                f"The field-name specified via `{_name}=<...>` is reserved by hydra-zen."
                " You can manually create a dataclass to utilize this name in a structured config."
            )

    target_field: List[Union[Tuple[str, Type[Any]], Tuple[str, Type[Any], Any]]]

    if (
        HYDRA_SUPPORTS_PARTIAL
        and zen_partial
        # check that no other zen-processing is needed
        and not zen_meta
        and not validated_wrappers
    ):  # pragma: no cover
        # TODO: require test-coverage once Hydra publishes nightly builds
        target_field = [
            (
                _TARGET_FIELD_NAME,
                str,
                _utils.field(default=_utils.get_obj_path(target), init=False),
            ),
            (
                _PARTIAL_FIELD_NAME,
                str,
                _utils.field(default=zen_partial, init=False),
            ),
        ]
    elif zen_partial or zen_meta or validated_wrappers:
        # target is `hydra_zen.funcs.zen_processing`
        target_field = [
            (
                _TARGET_FIELD_NAME,
                str,
                _utils.field(default=_utils.get_obj_path(zen_processing), init=False),
            ),
            (
                _ZEN_TARGET_FIELD_NAME,
                str,
                _utils.field(default=_utils.get_obj_path(target), init=False),
            ),
        ]

        if zen_partial:
            target_field.append(
                (
                    _ZEN_PARTIAL_TARGET_FIELD_NAME,
                    bool,
                    _utils.field(default=True, init=False),
                ),
            )

        if zen_meta:
            target_field.append(
                (
                    _META_FIELD_NAME,
                    Tuple[str, ...],
                    _utils.field(default=tuple(zen_meta), init=False),
                ),
            )

        if validated_wrappers:
            if zen_meta:
                # Check to see
                tuple(
                    _utils.check_suspicious_interpolations(
                        validated_wrappers, zen_meta=zen_meta, target=target
                    )
                )
            if len(validated_wrappers) == 1:
                # we flatten the config to avoid unnecessary list
                target_field.append(
                    (
                        _ZEN_WRAPPERS_FIELD_NAME,
                        Union[Union[str, Builds], Tuple[Union[str, Builds], ...]],
                        _utils.field(default=validated_wrappers[0], init=False),
                    ),  # type: ignore
                )
            else:
                target_field.append(
                    (
                        _ZEN_WRAPPERS_FIELD_NAME,
                        Union[Union[str, Builds], Tuple[Union[str, Builds], ...]],
                        _utils.field(default=validated_wrappers, init=False),
                    ),  # type: ignore
                )
    else:
        target_field = [
            (
                _TARGET_FIELD_NAME,
                str,
                _utils.field(default=_utils.get_obj_path(target), init=False),
            )
        ]

    base_fields = target_field

    if hydra_recursive is not None:
        base_fields.append(
            (
                _RECURSIVE_FIELD_NAME,
                bool,
                _utils.field(default=hydra_recursive, init=False),
            )
        )

    if hydra_convert is not None:
        base_fields.append(
            (_CONVERT_FIELD_NAME, str, _utils.field(default=hydra_convert, init=False))
        )

    if _pos_args:
        base_fields.append(
            (
                _POS_ARG_FIELD_NAME,
                Tuple[Any, ...],
                _utils.field(
                    default=tuple(
                        sanitized_default_value(x, error_prefix=BUILDS_ERROR_PREFIX)
                        for x in _pos_args
                    ),
                    init=False,
                ),
            )
        )

    try:
        # We want to rely on `inspect.signature` logic for raising
        # against an uninspectable sig, before we start inspecting
        # class-specific attributes below.
        signature_params = inspect.signature(target).parameters
    except ValueError:
        if populate_full_signature:
            raise ValueError(
                BUILDS_ERROR_PREFIX
                + f"{target} does not have an inspectable signature. "
                f"`builds({_utils.safe_name(target)}, populate_full_signature=True)` is not supported"
            )
        signature_params: Mapping[str, inspect.Parameter] = {}
        # We will turn off signature validation for objects that didn't have
        # a valid signature. This will enable us to do things like `build(dict, a=1)`
        target_has_valid_signature: bool = False
    else:
        # Dealing with this bug: https://bugs.python.org/issue40897
        #
        # In Python < 3.9.1, `inspect.signature will look first to
        # any implementation __new__, even if it is inherited and if
        # there is a "fresher" __init__.
        #
        # E.g. anything that inherits from `typing.Generic` and
        # does not implement its own __new__ will have a reported sig
        # of (*args, **kwargs)
        #
        # This looks specifically for the scenario that the target
        # has inherited from a parent that implements __new__ and
        # the target implements only __init__.
        if (
            inspect.isclass(target)
            and len(target.__mro__) > 2
            and "__init__" in target.__dict__
            and "__new__" not in target.__dict__
            and any("__new__" in parent.__dict__ for parent in target.__mro__[1:-1])
        ):
            _params = tuple(inspect.signature(target.__init__).parameters.items())

            if _params and _params[0][1].kind is not _VAR_POSITIONAL:
                # Exclude self/cls
                #
                # There are weird edge cases, like in collections.Counter for Python 3.7
                # where the first arg is *args, not self.
                _params = _params[1:]

            signature_params = {k: v for k, v in _params}
            del _params

        target_has_valid_signature: bool = True

    # `get_type_hints` properly resolves forward references, whereas annotations from
    # `inspect.signature` do not
    try:
        if inspect.isclass(target):
            # This implements the same method prioritization as
            # `inspect.signature` for Python >= 3.9.1
            if "__new__" in target.__dict__:
                _annotation_target = target.__new__
            elif "__init__" in target.__dict__:
                _annotation_target = target.__init__
            elif len(target.__mro__) > 2 and any(
                "__new__" in parent.__dict__ for parent in target.__mro__[1:-1]
            ):
                _annotation_target = target.__new__
            else:
                _annotation_target = target.__init__
        else:
            _annotation_target = target

        type_hints = get_type_hints(_annotation_target)

        del _annotation_target
        # We don't need to pop self/class because we only make on-demand
        # requests from `type_hints`

    except (
        TypeError,  # ufuncs, which do not have inspectable type hints
        NameError,  # Unresolvable forward reference
        AttributeError,  # Class doesn't have "__new__" or "__init__"
    ):
        type_hints = defaultdict(lambda: Any)

    sig_by_kind: Dict[Any, List[inspect.Parameter]] = {
        _POSITIONAL_ONLY: [],
        _POSITIONAL_OR_KEYWORD: [],
        _VAR_POSITIONAL: [],
        _KEYWORD_ONLY: [],
        _VAR_KEYWORD: [],
    }

    for p in signature_params.values():
        sig_by_kind[p.kind].append(p)

    # these are the names of the only parameters in the signature of `target` that can
    # be referenced by name
    nameable_params_in_sig: Set[str] = {
        p.name
        for p in chain(sig_by_kind[_POSITIONAL_OR_KEYWORD], sig_by_kind[_KEYWORD_ONLY])
    }

    if not _pos_args and builds_bases:
        # pos_args is potentially inherited
        for _base in builds_bases:
            _pos_args = getattr(_base, _POS_ARG_FIELD_NAME, ())

            # validates
            _pos_args = tuple(
                sanitized_default_value(x, allow_zen_conversion=False)
                for x in _pos_args
            )
            if _pos_args:
                break

    fields_set_by_bases: Set[str] = {
        _field.name
        for _base in builds_bases
        for _field in fields(_base)
        if _field.name not in _HYDRA_FIELD_NAMES and not _field.name.startswith("_zen_")
    }

    # Validate that user-specified arguments satisfy target's signature.
    # Should catch:
    #    - bad parameter names
    #    - too many parameters-by-position
    #    - multiple values specified for parameter (via positional and by-name)
    #
    # We don't raise on an under-specified signature because it is possible that the
    # resulting dataclass will simply be inherited from and extended.
    # The issues we catch here cannot be fixed downstream.
    if target_has_valid_signature:

        if not sig_by_kind[_VAR_KEYWORD]:
            # check for unexpected kwargs
            if not set(kwargs_for_target) <= nameable_params_in_sig:
                _unexpected = set(kwargs_for_target) - nameable_params_in_sig
                raise TypeError(
                    BUILDS_ERROR_PREFIX
                    + f"The following unexpected keyword argument(s) was specified for {_utils.get_obj_path(target)} "
                    f"via `builds`: {', '.join(_unexpected)}"
                )
            if not fields_set_by_bases <= nameable_params_in_sig and not (
                fields_set_by_bases - nameable_params_in_sig
            ) <= set(zen_meta):
                # field inherited by base is not present in sig
                # AND it is not excluded via `zen_meta`
                _unexpected = fields_set_by_bases - nameable_params_in_sig
                raise TypeError(
                    BUILDS_ERROR_PREFIX
                    + f"The following unexpected keyword argument(s) for {_utils.get_obj_path(target)} "
                    f"was specified via inheritance from a base class: "
                    f"{', '.join(_unexpected)}"
                )

        if _pos_args:
            named_args = set(kwargs_for_target).union(fields_set_by_bases)

            # indicates that number of parameters that could be specified by name,
            # but are specified by position
            _num_nameable_args_by_position = max(
                0, len(_pos_args) - len(sig_by_kind[_POSITIONAL_ONLY])
            )
            if named_args:
                # check for multiple values for arg, specified both via positional and kwarg
                # E.g.: def f(x, y): ...
                # f(1, 2, y=3)  # multiple values for `y`
                for param in sig_by_kind[_POSITIONAL_OR_KEYWORD][
                    :_num_nameable_args_by_position
                ]:
                    if param.name in named_args:
                        raise TypeError(
                            BUILDS_ERROR_PREFIX
                            + f"Multiple values for argument {param.name} were specified for "
                            f"{_utils.get_obj_path(target)} via `builds`"
                        )
            if not sig_by_kind[
                _VAR_POSITIONAL
            ] and _num_nameable_args_by_position > len(
                sig_by_kind[_POSITIONAL_OR_KEYWORD]
            ):
                # Too many positional args specified.
                # E.g.: def f(x, y): ...
                # f(1, 2, 3)
                _num_positional = len(sig_by_kind[_POSITIONAL_ONLY]) + len(
                    sig_by_kind[_POSITIONAL_OR_KEYWORD]
                )
                _num_with_default = sum(
                    p.default is not inspect.Parameter.empty
                    and p.kind is _POSITIONAL_OR_KEYWORD
                    for p in signature_params.values()
                )
                _permissible = (
                    f"{_num_positional}"
                    if not _num_with_default
                    else f"{_num_positional - _num_with_default} to {_num_positional}"
                )
                raise TypeError(
                    BUILDS_ERROR_PREFIX
                    + f"{_utils.get_obj_path(target)} takes {_permissible} positional args, but "
                    f"{len(_pos_args)} were specified via `builds`"
                )

    # Create valid dataclass fields from the user-specified values
    #
    # user_specified_params: arg-name -> (arg-name, arg-type, field-w-value)
    #  - arg-type: taken from the parameter's annotation in the target's signature
    #    and is resolved to one of the type annotations supported by hydra if possible,
    #    otherwise, is Any
    #  - arg-value: mutable values are automatically specified using default-factory
    user_specified_named_params: Dict[str, Tuple[str, type, Any]] = {
        name: (name, type_hints.get(name, Any), value)
        for name, value in kwargs_for_target.items()
    }

    if populate_full_signature is True:
        # Populate dataclass fields based on the target's signature.
        #
        # A user-specified parameter value (via `kwargs_for_target`) takes precedent over
        # the default value from the signature

        # Fields with default values must come after those without defaults,
        # so we will collect these as we loop through the parameters and
        # add them to the fields at the end.
        #
        # Parameter ordering should only differ from the target's signature
        # if the user specified a value for a parameter that had no default
        _fields_with_default_values: List[Field_Entry] = []

        # we need to keep track of what user-specified params we have set
        _seen: Set[str] = set()

        for n, param in enumerate(signature_params.values()):
            if n + 1 <= len(_pos_args):
                # Positional parameters are populated from "left to right" in the signature.
                # We have already done validation, so we know that positional params aren't redundant
                # with named params (including inherited params).
                continue

            if param.name not in nameable_params_in_sig:
                # parameter cannot be specified by name
                continue

            if param.name in user_specified_named_params:
                # user-specified parameter is preferred
                _fields_with_default_values.append(
                    user_specified_named_params[param.name]
                )
                _seen.add(param.name)
            else:
                # any parameter whose default value is None is automatically
                # annotated with `Optional[...]`. This improves flexibility with
                # Hydra's type-validation
                param_field = (
                    param.name,
                    type_hints.get(param.name, Any),
                )

                if param.default is inspect.Parameter.empty:
                    if not zen_partial:
                        # No default value specified in signature or by the user.
                        # We don't include these fields if the user specified a partial build
                        # because we assume that they want to fill these in by using partial
                        base_fields.append(param_field)
                else:
                    param_field += (param.default,)
                    _fields_with_default_values.append(param_field)

        base_fields.extend(_fields_with_default_values)

        if sig_by_kind[_VAR_KEYWORD]:
            # if the signature has **kwargs, then we need to add any user-specified
            # parameters that have not already been added
            base_fields.extend(
                entry
                for name, entry in user_specified_named_params.items()
                if name not in _seen
            )
    else:
        base_fields.extend(user_specified_named_params.values())

    if zen_meta:
        _meta_names = set(zen_meta)

        if _meta_names & nameable_params_in_sig:
            raise ValueError(
                f"`builds(..., zen_meta=<...>)`: `zen_meta` cannot not specify "
                f"names that exist in the target's signature: "
                f"{','.join(_meta_names & nameable_params_in_sig)}"
            )

        if _meta_names & set(user_specified_named_params):
            raise ValueError(
                f"`builds(..., zen_meta=<...>)`: `zen_meta` cannot not specify "
                f"names that are common with those specified in **kwargs_for_target: "
                f"{','.join(_meta_names & set(user_specified_named_params))}"
            )

        # We don't check for collisions between `zen_meta` names and the
        # names of inherited fields. Thus `zen_meta` can effectively be used
        # to "delete" names from a config, via inheritance.
        base_fields.extend((name, Any, value) for name, value in zen_meta.items())

    if dataclass_name is None:
        if zen_partial is False:
            dataclass_name = f"Builds_{_utils.safe_name(target)}"
        else:
            dataclass_name = f"PartialBuilds_{_utils.safe_name(target)}"

    # validate that fields set via bases are OK; cannot perform zen-casting
    # on fields
    for base in builds_bases:
        for field_ in fields(base):
            if field_.default is not MISSING:
                # performs validation
                sanitized_default_value(
                    field_.default,
                    allow_zen_conversion=False,
                    error_prefix=BUILDS_ERROR_PREFIX,
                    field_name=field_.name + " (set via inheritance)",
                )
            del field_

    # sanitize all types and configured values
    sanitized_base_fields: List[Union[Tuple[str, Any], Tuple[str, Any, Field]]] = []

    for item in base_fields:
        name = item[0]
        type_ = item[1]
        if len(item) == 2:
            sanitized_base_fields.append((name, _utils.sanitized_type(type_)))
        else:
            assert len(item) == 3, item
            value = item[-1]

            if not isinstance(value, Field):
                _field = sanitized_field(
                    value,
                    error_prefix=BUILDS_ERROR_PREFIX,
                    field_name=item[0],
                    _mutable_default_permitted=_utils.mutable_default_permitted(
                        builds_bases, name
                    ),
                )
            elif (
                PATCH_OMEGACONF_830
                and builds_bases
                and value.default_factory is not MISSING
            ):

                # Addresses omegaconf #830 https://github.com/omry/omegaconf/issues/830
                #
                # Value was passed as a field-with-default-factory, we'll
                # access the default from the factory and will reconstruct the field
                _field = sanitized_field(
                    value.default_factory(),
                    error_prefix=BUILDS_ERROR_PREFIX,
                    field_name=item[0],
                    _mutable_default_permitted=_utils.mutable_default_permitted(
                        builds_bases, name
                    ),
                )
            else:
                _field = value

            # If `.default` is not set, then `value` is a Hydra-supported mutable
            # value, and thus it is "sanitized"
            sanitized_value = getattr(_field, "default", value)
            sanitized_type = (
                _utils.sanitized_type(type_, wrap_optional=sanitized_value is None)
                # OmegaConf's type-checking occurs before instantiation occurs.
                # This means that, e.g., passing `Builds[int]` to a field `x: int`
                # will fail Hydra's type-checking upon instantiation, even though
                # the recursive instantiation will appropriately produce `int` for
                # that field. This will not be addressed by hydra/omegaconf:
                #    https://github.com/facebookresearch/hydra/issues/1759
                # Thus we will auto-broaden the annotation when we see that a field
                # is set with a structured config as a default value - assuming that
                # the field isn't annotated with a structured config type.
                if hydra_recursive is False
                or not is_builds(sanitized_value)
                or is_builds(type_)
                else Any
            )
            sanitized_base_fields.append((name, sanitized_type, _field))
            del value
            del _field
            del sanitized_value

    out = make_dataclass(
        dataclass_name, fields=sanitized_base_fields, bases=builds_bases, frozen=frozen
    )

    if zen_partial is False and hasattr(out, _ZEN_PARTIAL_TARGET_FIELD_NAME):
        # `out._partial_target_` has been inherited; this will lead to an error when
        # hydra-instantiation occurs, since it will be passed to target.
        # There is not an easy way to delete this, since it comes from a parent class
        raise TypeError(
            BUILDS_ERROR_PREFIX
            + "`builds(..., zen_partial=False, builds_bases=(...))` does not "
            "permit `builds_bases` where a partial target has been specified."
        )

    out.__doc__ = (
        f"A structured config designed to {'partially ' if zen_partial else ''}initialize/call "
        f"`{_utils.get_obj_path(target)}` upon instantiation by hydra."
    )
    if hasattr(target, "__doc__"):
        target_doc = target.__doc__
        if target_doc:
            out.__doc__ += (
                f"\n\nThe docstring for {_utils.safe_name(target)} :\n\n" + target_doc
            )
    return cast(Type[Builds[Importable]], out)


# We need to check if things are Builds, Just, PartialBuilds to a higher
# fidelity than is provided by `isinstance(..., <Protocol>)`. I.e. we want to
# check that the desired attributes *and* that their values match those of the
# protocols. Failing to heed this would, for example, lead to any `Builds` that
# happens to have a `path` attribute to be treated as `Just` in `get_target`.
#
# The following functions perform these desired checks. Note that they do not
# require that the provided object be a dataclass; this enables compatibility
# with omegaconf containers.
#
# These are not part of the public API for now, but they may be in the future.
def is_builds(x: Any) -> TypeGuard[Builds]:
    return hasattr(x, _TARGET_FIELD_NAME)


def is_just(x: Any) -> TypeGuard[Just]:
    if is_builds(x) and hasattr(x, _JUST_FIELD_NAME):
        attr = _get_target(x)
        if attr == _get_target(Just) or attr is get_obj:
            return True
        else:
            # ensures we conver this branch in tests
            return False
    return False


def _is_old_partial_builds(x: Any) -> bool:  # pragma: no cover
    # We don't care about coverage here.
    # This will only be used in `get_target` and we'll be sure to cover that branch
    if is_builds(x) and hasattr(x, "_partial_target_"):
        attr = _get_target(x)
        if (attr == "hydra_zen.funcs.partial" or attr is partial) and is_just(
            getattr(x, "_partial_target_")
        ):
            return True
        else:
            # ensures we cover this branch in tests
            return False
    return False


def uses_zen_processing(x: Any) -> TypeGuard[Builds]:
    if not is_builds(x) or not hasattr(x, _ZEN_TARGET_FIELD_NAME):
        return False
    attr = _get_target(x)
    if attr != _ZEN_PROCESSING_LOCATION and attr is not zen_processing:
        return False
    return True


def is_partial_builds(x: Any) -> TypeGuard[PartialBuilds]:
    return (
        # check if partial'd config via Hydra
        HYDRA_SUPPORTS_PARTIAL
        and getattr(x, _PARTIAL_FIELD_NAME, False) is True
    ) or (
        # check if partial'd config via `zen_processing`
        uses_zen_processing(x)
        and (getattr(x, _ZEN_PARTIAL_TARGET_FIELD_NAME, False) is True)
    )


@overload
def get_target(
    obj: Union[PartialBuilds[_T], Type[PartialBuilds[_T]]]
) -> _T:  # pragma: no cover
    ...


@overload
def get_target(obj: Union[Just[_T], Type[Just[_T]]]) -> _T:  # pragma: no cover
    ...


@overload
def get_target(obj: Union[Builds[_T], Type[Builds[_T]]]) -> _T:  # pragma: no cover
    ...


@overload
def get_target(obj: Union[HasTarget, HasPartialTarget]) -> Any:  # pragma: no cover
    ...


def get_target(obj: Union[HasTarget, HasPartialTarget]) -> Any:
    """
    Returns the target-object from a targeted config.

    Parameters
    ----------
    obj : HasTarget
        An object with a ``_target_`` attribute.

    Returns
    -------
    target : Any

    Raises
    ------
    TypeError: ``obj`` does not have a ``_target_`` attribute.

    Examples
    --------
    **Basic usage**

    `get_target` works on all variety of configs produced by `builds` and
    `just`.

    >>> from hydra_zen import builds, just, get_target
    >>> get_target(builds(int))
    <class 'int'>
    >>> get_target(builds(int, zen_partial=True))
    <class 'int'>
    >>> get_target(just(str))
    <class 'str'>

    It works for manually-defined configs:

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class A:
    ...     _target_: str = "builtins.dict"

    >>> get_target(A)
    <class 'dict'>

    and for configs loaded from yamls.

    >>> from hydra_zen import load_from_yaml, save_as_yaml

    >>> class B: pass
    >>> Conf = builds(B)

    >>> save_as_yaml(Conf, "config.yaml")
    >>> loaded_conf = load_from_yaml("config.yaml")

    Note that the target of ``loaded_conf`` can be accessed without
    instantiating the config.

    >>> get_target(loaded_conf)
    __main__.B
    """
    if _is_old_partial_builds(obj):
        # obj._partial_target_ is `Just[obj]`
        return get_target(getattr(obj, "_partial_target_"))
    elif uses_zen_processing(obj):
        field_name = _ZEN_TARGET_FIELD_NAME
    elif is_just(obj):
        field_name = _JUST_FIELD_NAME
    elif is_builds(obj):
        field_name = _TARGET_FIELD_NAME
    else:
        raise TypeError(
            f"`obj` must specify a target; i.e. it must have an attribute named"
            f" {_TARGET_FIELD_NAME} or named {_ZEN_PARTIAL_TARGET_FIELD_NAME} that"
            f" points to a target-object or target-string"
        )
    target = getattr(obj, field_name)

    if isinstance(target, str):
        target = get_obj(path=target)
    else:
        # Hydra 1.1.0 permits objects-as-_target_ instead of strings
        # https://github.com/facebookresearch/hydra/issues/1017
        pass  # makes sure we cover this branch in tests

    return target


_builds_sig = inspect.signature(builds)
__BUILDS_DEFAULTS: Final[Dict[str, Any]] = {
    name: p.default
    for name, p in _builds_sig.parameters.items()
    if p.kind is p.KEYWORD_ONLY
}
del _builds_sig


def make_custom_builds_fn(
    *,
    zen_partial: bool = False,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    builds_bases: Tuple[Any, ...] = (),
    # This is the easiest way to get static tooling to see the output
    # as `builds`.
    __b: _T2 = builds,
) -> _T2:
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

    hydra_recursive : Optional[bool], optional (default=True)
        Specifies a new the default value for ``builds(..., hydra_recursive=<..>)``

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Specifies a new the default value for ``builds(..., hydra_convert=<..>)``

    frozen : bool, optional (default=False)
        Specifies a new the default value for ``builds(..., frozen=<..>)``

    builds_bases : Tuple[DataClass, ...]
        Specifies a new the default value for ``builds(..., builds_bases=<..>)``

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
    if __b is not builds:
        raise TypeError("make_custom_builds_fn() got an unexpected argument: '__b'")
    del __b

    excluded_fields = {"dataclass_name"}
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
    def wrapped(*args, **kwargs):
        merged_kwargs = {}
        merged_kwargs.update(_new_defaults)
        merged_kwargs.update(kwargs)
        return builds(*args, **merged_kwargs)

    return cast(_T2, wrapped)


class NOTHING:
    def __init__(self) -> None:
        raise TypeError("`NOTHING` cannot be instantiated")


@dataclass
class ZenField:
    """
    ZenField(hint=Any, default=<class 'NOTHING'>, name=<class 'NOTHING'>)

    Specifies a field's name and/or type-annotation and/or default value.
    Designed to specify fields in `make_config`.

    See the Examples section of the docstring for `make_config` for examples of using
    `ZenField`.

    Parameters
    ----------
    hint : type, optional (default=Any)
    default : Any, optional
    name : str, optional

    Notes
    -----
    ``default`` will be returned as an instance of :class:`dataclasses.Field`.
    Mutable values (e.g. lists or dictionaries) passed to ``default`` will automatically
    be "packaged" in a default-factory function [1]_.

    A type passed to ``hint`` will automatically be "broadened" such that the resulting
    type is compatible with Hydra's set of supported type annotations [2]_.

    References
    ----------
    .. [1] https://docs.python.org/3/library/dataclasses.html#default-factory-functions
    .. [2] https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-supports

    See Also
    --------
    make_config: create a config with customized field names, default values, and annotations.
    """

    hint: type = Any
    default: Union[SupportedPrimitive, Field, Type[NOTHING]] = NOTHING
    name: Union[str, Type[NOTHING]] = NOTHING
    _permit_default_factory: InitVar[bool] = True

    def __post_init__(self, _permit_default_factory):
        if not isinstance(self.name, str):
            if self.name is not NOTHING:
                raise TypeError(f"`ZenField.name` expects a string, got: {self.name}")

        self.hint = _utils.sanitized_type(self.hint)

        if self.default is not NOTHING:
            self.default = sanitized_field(
                self.default, _mutable_default_permitted=_permit_default_factory
            )


def make_config(
    *fields_as_args: Union[str, ZenField],
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    config_name: str = "Config",
    frozen: bool = False,
    bases: Tuple[Type[_DataClass], ...] = (),
    **fields_as_kwargs: Union[SupportedPrimitive, ZenField],
) -> Type[DataClass]:
    """
    Creates a config with user-defined field names and, optionally,
    associated default values and/or type annotations.

    Unlike `builds`, `make_config` is not used to configure a particular target
    object; rather, it can be used to create more general configs [1]_.

    Parameters
    ----------
    *fields_as_args : str | ZenField
        The names of the fields to be be included in the config. Or, `ZenField`
        instances, each of which details the name and their default value and/or the
        type annotation of a given field.

    **fields_as_kwargs : SupportedPrimitive | ZenField
        Like ``fields_as_args``, but field-name/default-value pairs are
        specified as keyword arguments. `ZenField` can also be used here
        to express a field's type-annotation and/or its default value.

        Named parameters of the forms ``hydra_xx``, ``zen_xx``, and ``_zen_xx`` are reserved to ensure future-compatibility, and cannot be specified by the user.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive objects passed to configuration [3]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    bases : Tuple[Type[DataClass], ...], optional (default=())
        Base classes that the resulting config class will inherit from.

    frozen : bool, optional (default=False)
        If ``True``, the resulting config class will produce 'frozen' (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance of the config will raise
        :py:class:`dataclasses.FrozenInstanceError` at runtime.

    config_name : str, optional (default="Config")
        The class name of the resulting config class.

    Returns
    -------
    Config : Type[DataClass]
        The resulting config class; a dataclass that possess the user-specified fields.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError
        The provided configured value cannot be serialized by Hydra, nor does hydra-zen
        provide specialized support for it. See :ref:`valid-types` for more details.

    Notes
    -----
    The resulting "config" is a dataclass-object [4]_ with Hydra-specific attributes
    attached to it, along with the attributes specified via ``fields_as_args`` and
    ``fields_as_kwargs``.

    Any field specified without a type-annotation is automatically annotated with
    :py:class:`typing.Any`. Hydra only supports a narrow subset of types [5]_;
    `make_config` will automatically 'broaden' any user-specified annotations so that
    they are compatible with Hydra.

    `make_config` will automatically manipulate certain types of default values to
    ensure that they can be utilized in the resulting config and by Hydra:

    - Mutable default values will automatically be packaged in a default factory function [6]_
    - A default value that is a class-object or function-object will automatically be wrapped by `just`, to ensure that the resulting config is serializable by Hydra.

    For finer-grain control over how type annotations and default values are managed,
    consider using :func:`dataclasses.make_dataclass`.

    For details of the annotation `SupportedPrimitive`, see :ref:`valid-types`.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    just : Create a config that "just" returns a class-object or function, without instantiating/calling it.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [4] https://docs.python.org/3/library/dataclasses.html
    .. [5] https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-supports
    .. [6] https://docs.python.org/3/library/dataclasses.html#default-factory-functions

    Examples
    --------
    >>> from hydra_zen import make_config, to_yaml
    >>> def pp(x):
    ...     return print(to_yaml(x))  # pretty-print config as yaml

    **Basic Usage**

    Let's create a bare-bones config with two fields, named 'a' and 'b'.

    >>> Conf1 = make_config("a", "b")  # sig: `Conf(a: Any, b: Any)`
    >>> pp(Conf1)
    a: ???
    b: ???

    Now we'll configure these fields with particular values:

    >>> conf1 = Conf1(1, "hi")
    >>> pp(conf1)
    a: 1
    b: hi
    >>> conf1.a
    1
    >>> conf1.b
    'hi'

    We can also specify fields via keyword args; this is especially convenient
    for providing associated default values.

    >>> Conf2 = make_config("unit", data=[-10, -20])
    >>> pp(Conf2)
    unit: ???
    data:
    - -10
    - -20

    Configurations can be nested

    >>> Conf3 = make_config(c1=Conf1(a=1, b=2), c2=Conf2)
    >>> pp(Conf3)
    c1:
      a: 1
      b: 2
    c2:
      unit: ???
      data:
      - -10
      - -20
    >>> Conf3.c1.a
    1

    Configurations can be composed via inheritance

    >>> ConfInherit = make_config(c=2, bases=(Conf2, Conf1))
    >>> pp(ConfInherit)
    a: ???
    b: ???
    unit: ???
    data:
    - -10
    - -20
    c: 2

    >>> issubclass(ConfInherit, Conf1) and issubclass(ConfInherit, Conf2)
    True

    **Support for Additional Types**

    Types like :py:class:`complex` and :py:class:`pathlib.Path` are automatically
    supported by hydra-zen.

    >>> ConfWithComplex = make_config(a=1+2j)
    >>> pp(ConfWithComplex)
    a:
      real: 1.0
      imag: 2.0
      _target_: builtins.complex

    See :ref:`additional-types` for a complete list of supported types.

    **Using ZenField to Provide Type Information**

    The `ZenField` class can be used to include a type-annotation in association
    with a field.

    >>> from hydra_zen import ZenField as zf
    >>> ProfileConf = make_config(username=zf(str), age=zf(int))
    >>> # signature: ProfileConf(username: str, age: int)

    Providing type annotations is optional, but doing so enables Hydra to perform
    checks at runtime to ensure that a configured value matches its associated
    type [4]_.

    >>> pp(ProfileConf(username="piro", age=False))  # age should be an integer
    <ValidationError: Value 'False' could not be converted to Integer>

    These default values can be provided alongside type annotations

    >>> C = make_config(age=zf(int, 0))  # signature: C(age: int = 0)

    `ZenField` can also be used to specify ``fields_as_args``; here, field names
    must be specified as well.

    >>> C2 = make_config(zf(name="username", hint=str), age=zf(int, 0))
    >>> # signature: C2(username: str, age: int = 0)

    See :ref:`data-val` for more general data validation capabilities via hydra-zen.
    """
    for _field in fields_as_args:
        if not isinstance(_field, (str, ZenField)):
            raise TypeError(
                f"`fields_as_args` can only consist of field-names (i.e. strings) or "
                f"`ZenField` instances. Got: "
                f"{', '.join(str(x) for x in fields_as_args if not isinstance(x, (str, ZenField)))}"
            )
        if isinstance(_field, ZenField) and _field.name is NOTHING:
            raise ValueError(
                f"All `ZenField` instances specified in `fields_as_args` must have a "
                f"name associated with it. Got: {_field}"
            )
    for name, _field in fields_as_kwargs.items():
        if isinstance(_field, ZenField):
            if _field.name is not NOTHING and _field.name != name:
                raise ValueError(
                    f"`fields_as_kwargs` specifies conflicting names: the kwarg {name} "
                    f"is associated with a `ZenField` with name {_field.name}"
                )
            else:
                _field.name = name

    if fields_as_args:
        all_names = [f.name if isinstance(f, ZenField) else f for f in fields_as_args]
        all_names.extend(fields_as_kwargs)

        if len(all_names) != len(set(all_names)):
            raise ValueError(
                f"`fields_as_args` cannot specify the same field-name multiple times."
                f" Got multiple entries for:"
                f" {', '.join(str(n) for n, count in Counter(all_names).items() if count > 1)}"
            )
        for _name in all_names:
            if isinstance(_name, str) and _name.startswith("_zen_"):
                raise ValueError(
                    f"The field-name specified via `{_name}=<...>` is reserved by hydra-zen."
                    " You can manually create a dataclass to utilize this name in a structured config."
                )
        del all_names

    # validate hydra-args via `builds`
    # also check for use of reserved names
    builds(
        dict,
        hydra_convert=hydra_convert,
        hydra_recursive=hydra_recursive,
        **fields_as_kwargs,
    )

    normalized_fields: Dict[str, ZenField] = {}

    for _field in fields_as_args:
        if isinstance(_field, str):
            normalized_fields[_field] = ZenField(name=_field, hint=Any)
        else:
            assert isinstance(_field.name, str)
            normalized_fields[_field.name] = _repack_zenfield(
                _field, _field.name, bases
            )

    for name, value in fields_as_kwargs.items():
        if not isinstance(value, ZenField):
            default_factory_permitted = (
                not bases or _utils.mutable_default_permitted(bases, field_name=name)
                if PATCH_OMEGACONF_830
                else True
            )
            normalized_fields[name] = ZenField(
                name=name,
                default=value,
                _permit_default_factory=default_factory_permitted,
            )
        else:
            normalized_fields[name] = _repack_zenfield(value, name=name, bases=bases)

    # fields without defaults must come first
    config_fields: List[Union[Tuple[str, type], Tuple[str, type, Any]]] = [
        (str(f.name), f.hint)
        for f in normalized_fields.values()
        if f.default is NOTHING
    ]

    config_fields.extend(
        [
            (
                str(f.name),
                (
                    # f.default: Field
                    # f.default.default: Any
                    f.hint
                    if hydra_recursive is False
                    or not is_builds(f.default.default)  # type: ignore
                    or is_builds(f.hint)
                    else Any
                ),
                f.default,
            )
            for f in normalized_fields.values()
            if f.default is not NOTHING
        ]
    )

    if hydra_recursive is not None:
        config_fields.append(
            (
                _RECURSIVE_FIELD_NAME,
                bool,
                _utils.field(default=hydra_recursive, init=False),
            )
        )

    if hydra_convert is not None:
        config_fields.append(
            (_CONVERT_FIELD_NAME, str, _utils.field(default=hydra_convert, init=False))
        )

    return cast(
        Type[DataClass],
        make_dataclass(
            cls_name=config_name, fields=config_fields, frozen=frozen, bases=bases
        ),
    )


def _repack_zenfield(value: ZenField, name: str, bases: Tuple[_DataClass, ...]):
    default = value.default

    if (
        PATCH_OMEGACONF_830
        and bases
        and not _utils.mutable_default_permitted(bases, field_name=name)
        and isinstance(default, Field)
        and default.default_factory is not MISSING
    ):
        return ZenField(
            hint=value.hint,
            default=default.default_factory(),
            name=value.name,
            _permit_default_factory=False,
        )
    return value


# registering value-conversions that depend on `builds`
def _cast_via_tuple(dest_type: Type[_T]) -> Callable[[_T], Type[Builds[Type[_T]]]]:
    def converter(value):
        return builds(dest_type, tuple(value))

    return converter


def _unpack_partial(value: Partial[_T]) -> Type[PartialBuilds[Type[_T]]]:
    target = cast(Type[_T], value.func)
    return builds(target, *value.args, **value.keywords, zen_partial=True)


ZEN_VALUE_CONVERSION[set] = _cast_via_tuple(set)
ZEN_VALUE_CONVERSION[frozenset] = _cast_via_tuple(frozenset)
ZEN_VALUE_CONVERSION[deque] = _cast_via_tuple(deque)
ZEN_VALUE_CONVERSION[bytes] = _cast_via_tuple(bytes)
ZEN_VALUE_CONVERSION[bytearray] = _cast_via_tuple(bytearray)
ZEN_VALUE_CONVERSION[range] = lambda value: builds(
    range, value.start, value.stop, value.step
)
ZEN_VALUE_CONVERSION[Counter] = lambda counter: builds(Counter, dict(counter))
ZEN_VALUE_CONVERSION[functools.partial] = _unpack_partial
