# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import warnings
from collections import Counter, defaultdict
from dataclasses import Field, dataclass, field, fields, is_dataclass, make_dataclass
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

from typing_extensions import Final, Literal, TypeGuard

from hydra_zen.errors import HydraZenDeprecationWarning
from hydra_zen.funcs import get_obj, partial, zen_processing
from hydra_zen.structured_configs import _utils
from hydra_zen.typing import Builds, Importable, Just, PartialBuilds
from hydra_zen.typing._implementations import (
    DataClass,
    HasPartialTarget,
    HasTarget,
    _DataClass,
)

try:
    # used to check if default values are ufuncs
    from numpy import ufunc  # type: ignore
except ImportError:  # pragma: no cover
    ufunc = None

_T = TypeVar("_T")
_T2 = TypeVar("_T2", bound=Callable)
ZenWrapper = Union[
    None,
    Builds[Callable[[_T2], _T2]],
    PartialBuilds[Callable[[_T2], _T2]],
    Just[Callable[[_T2], _T2]],
    Callable[[_T2], _T2],
    str,
]
if TYPE_CHECKING:  # pragma: no cover
    ZenWrappers = Union[ZenWrapper, Sequence[ZenWrapper]]
else:
    ZenWrappers = TypeVar("ZenWrappers")

# Hydra-specific fields
_TARGET_FIELD_NAME: Final[str] = "_target_"
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
_PARTIAL_TARGET_FIELD_NAME: Final[str] = "_zen_partial"
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
                    "as of 2021-09-18. Change `builds(target=<target>, ...)` to `builds(<target>, ...)`."
                    "\n\nThis will be an error in hydra-zen 1.0.0, or by 2021-12-18 — whichever "
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
                    "The argument `hydra_partial` is deprecated as of 2021-10-10.\n"
                    "Change `builds(..., hydra_partial=<..>)` to `builds(..., zen_partial=<..>)`."
                    "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-10 — whichever "
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

    >>> @dataclass
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
    *pos_args: Any,
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
    hydra-specific fields for specifying a structured config [1]_.

    Parameters
    ----------
    target : Union[Instantiable, Callable]
        The object to be instantiated/called.

    *pos_args : Any
        Positional arguments passed to ``target``.

        Arguments specified positionally are not included in the dataclass' signature and
        are stored as a tuple bound to in the ``_args_`` field.

    zen_partial : Optional[bool] (default=False)
        If True, then hydra-instantiation produces ``functools.partial(target, **kwargs)``

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(hydra_target)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [2]_ or targeted structured
        configs.

    zen_meta : Optional[Mapping[str, Any]]
        Specifies field-names and corresponding values that will be included in the
        resulting dataclass, but that will *not* be used to build ``hydra_target``
        via instantiation. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting dataclass's signature and fields will be populated
        according to the signature of ``hydra_target``.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_recursive : bool, optional (default=True)
        If True, then upon hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [3]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert : Optional[Literal["none", "partial", "all"]] (default="none")
        Determines how hydra handles the non-primitive objects passed to `target` [4]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with
          the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without
          a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    frozen : bool, optional (default=False)
        If `True`, the resulting dataclass will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise `FrozenInstanceError`
        at runtime.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [4] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    Examples
    --------
    A simple usage of `hydrated_dataclass`. Here, we specify a structured config

    >>> from hydra_zen import hydrated_dataclass, instantiate
    >>> @hydrated_dataclass(target=dict)
    ... class DictConf:
    ...     x: int = 2
    ...     y: str = 'hello'

    >>> instantiate(DictConf(x=10))  # override default `x`
    {'x': 10, 'y': 'hello'}

    We can also design a configuration that only partially instantiates our target.

    >>> def power(x: float, exponent: float) -> float: return x ** exponent
    >>> @hydrated_dataclass(target=power, zen_partial=True)
    ... class PowerConf:
    ...     exponent : float = 2.0

    >>> partiald_power = instantiate(PowerConf)
    >>> partiald_power(10.0)
    100.0

    Inheritance can be used to compose configurations

    >>> from dataclasses import dataclass
    >>> from torch.optim import AdamW
    >>> @dataclass
    ... class AdamBaseConfig:
    ...     lr: float = 0.001
    ...     eps: float = 1e-8

    >>> @hydrated_dataclass(target=AdamW, zen_partial=True)
    ... class AdamWConfig(AdamBaseConfig):
    ...     weight_decay : float = 0.01
    >>> instantiate(AdamWConfig)
    functools.partial(<class 'torch.optim.adamw.AdamW'>, lr=0.001, eps=1e-08, weight_decay=0.01)

    Because this decorator uses `hyda_utils.builds` under the hood, common mistakes like misspelled
    parameters will be caught upon constructing the structured config.

    >>> @hydrated_dataclass(target=AdamW, zen_partial=True)
    ... class AdamWConfig(AdamBaseConfig):
    ...     wieght_decay : float = 0.01  # i before e, right!?
    TypeError: Building: AdamW ..
    The following unexpected keyword argument(s) for torch.optim.adamw.AdamW was specified via inheritance
    from a base class: wieght_decay

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
                "The argument `hydra_partial` is deprecated as of 2021-10-10.\n"
                "Change `builds(..., hydra_partial=<..>)` to `builds(..., zen_partial=<..>)`."
                "\n\nThis will be an error in hydra-zen 1.0.0, or by 2022-01-10 — whichever "
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

        return builds(
            target,
            *pos_args,
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
    """Produces a structured config that, when instantiated by Hydra, 'just'
    returns the target (uninstantiated).

    This is convenient for specifying a particular, un-instantiated object as part of your
    configuration.

    Parameters
    ----------
    obj : Importable
        The object that will be instantiated from this config.

    Returns
    -------
    Type[Just[Importable]]
        The dataclass object that is designed as a structured config.

    Notes
    -----
    The configs produced by `just` introduce an explicit dependency on hydra-zen. I.e.
    hydra-zen must be installed in order to instantiate the config.

    Examples
    --------
    >>> from hydra_zen import just, instantiate, to_yaml
    >>> just_range = just(range)
    >>> range is instantiate(just_range)
    True
    >>> just_range._target_
    'hydra_zen.funcs.get_obj'
    >>> just_range.path
    'builtins.range'
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


def create_just_if_needed(value: _T) -> Union[_T, Type[Just]]:
    # Hydra can serialize dataclasses directly, thus we
    # don't want to wrap these in `just`

    if callable(value) and (
        inspect.isfunction(value)
        or (inspect.isclass(value) and not is_dataclass(value))
        or isinstance(value, _builtin_function_or_method_type)
        or (ufunc is not None and isinstance(value, ufunc))
    ):
        return just(value)

    return value


def sanitized_default_value(value: Any) -> Field:
    if isinstance(value, _utils.KNOWN_MUTABLE_TYPES):
        return cast(Field, mutable_value(value))
    resolved_value = create_just_if_needed(value)
    return (
        _utils.field(default=value)
        if value is resolved_value
        else _utils.field(default=resolved_value)
    )


# overloads when `zen_partial=False`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: Any,
    zen_partial: Literal[False] = False,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
) -> Type[Builds[Importable]]:  # pragma: no cover
    ...


# overloads when `zen_partial=True`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: Any,
    zen_partial: Literal[True],
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
) -> Type[PartialBuilds[Importable]]:  # pragma: no cover
    ...


# overloads when `zen_partial: bool`
@overload
def builds(
    hydra_target: Importable,
    *pos_args: Any,
    zen_partial: bool,
    zen_wrappers: ZenWrappers = tuple(),
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    dataclass_name: Optional[str] = None,
    builds_bases: Tuple[Any, ...] = (),
    frozen: bool = False,
    **kwargs_for_target,
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
    zen_meta: Optional[Mapping[str, Any]] = None,
    populate_full_signature: bool = False,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    frozen: bool = False,
    builds_bases: Tuple[Any, ...] = (),
    dataclass_name: Optional[str] = None,
    **kwargs_for_target,
) -> Union[Type[Builds[Importable]], Type[PartialBuilds[Importable]]]:
    """builds(hydra_target, /, *pos_args, zen_partial=False, zen_meta=None, hydra_recursive=None, populate_full_signature=False, hydra_convert=None, frozen=False, dataclass_name=None, builds_bases=(), **kwargs_for_target)

    Returns a dataclass object that configures ``<hydra_target>`` with user-specified and auto-populated parameter
    values.

    The resulting dataclass is specifically a structured config [1]_ that enables Hydra to initialize/call
    `target` either fully or partially. See Notes for additional features and explanation of implementation details.

    Parameters
    ----------
    hydra_target : T (Callable)
        The object to be configured. This is a required, positional-only argument.

    *pos_args : Any
        Positional arguments passed to ``hydra_target``.

        Arguments specified positionally are not included in the dataclass' signature and
        are stored as a tuple bound to in the ``_args_`` field.

    **kwargs_for_target : Any
        The keyword arguments passed to ``hydra_target(...)``.

        The arguments specified here solely determine the fields and init-parameters of the
        resulting dataclass, unless ``populate_full_signature=True`` is specified (see below).

        Named parameters of the forms ``hydra_xx``, ``zen_xx``, and ``_zen_xx`` are reserved
        to ensure future-compatibility, and cannot be specified by the user.

    zen_partial : bool, optional (default=False)
        If True, then the resulting config will instantiate as
        ``functools.partial(hydra_target, *pos_args, **kwargs_for_target)`` rather than
        ``hydra_target(*pos_args, **kwargs_for_target)``.

        Thus this enables the partial-configuration of objects.

        Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together will
        populate the dataclass' signature only with parameters that are specified by the
        user or that have default values specified in the target's signature. I.e. it is
        presumed that un-specified parameters are to be excluded from the partial configuration.

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(hydra_target)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [2]_ or targeted structured
        configs.

    zen_meta : Optional[Mapping[str, Any]]
        Specifies field-names and corresponding values that will be included in the
        resulting dataclass, but that will *not* be used to build ``hydra_target``
        via instantiation. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting dataclass's signature and fields will be populated
        according to the signature of ``hydra_target``.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [3]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive objects passed to `target` [4]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with
          the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without
          a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    frozen : bool, optional (default=False)
        If ``True``, the resulting dataclass will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise ``FrozenInstanceError``
        at runtime.

    builds_bases : Tuple[DataClass, ...]
        Specifies a tuple of parent classes that the resulting dataclass inherits from.
        A ``PartialBuilds`` class (resulting from ``zen_partial=True``) cannot be a parent
        of a ``Builds`` class (i.e. where `zen_partial=False` was specified).

    dataclass_name : Optional[str]
        If specified, determines the name of the returned class object.

    Returns
    -------
    Config : Type[Builds[Type[T]]] |  Type[PartialBuilds[Type[T]]]
        A structured config that builds ``hydra_target``

    Notes
    -----
    Using any of the `zen_xx` features will result in a config that depends
    explicitly on hydra-zen. (i.e. hydra-zen must be installed in order to
    instantiate the resulting config, including its yaml version).

    Type annotations are inferred from the target's signature and are only
    retained if they are compatible with hydra's limited set of supported
    annotations; otherwise `Any` is specified.

    `builds` provides runtime validation of user-specified named arguments against
    the target's signature. This helps to ensure that typos in field names
    fail early and explicitly.

    Mutable values are automatically transformed to use a default factory [5]_
    prior to setting them on the dataclass.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [4] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [5] https://docs.python.org/3/library/dataclasses.html#mutable-default-values

    See Also
    --------
    instantiate: Instantiates a configuration created by `builds`.
    make_custom_builds_fn: Returns the `builds` function, but with customized default values.
    make_config: Creates a config with customized field names, default values, and annotations.
    get_target: Returns the target-object from a targeted structured config.
    just: Produces a config that, when instantiated by Hydra, "just" returns the uninstantiated target.

    Examples
    --------
    Basic Usage:

    >>> from hydra_zen import builds, instantiate
    >>> Conf = builds(dict, a=1, b='x')  # makes a dataclass that will "build" a dictionary with the specified fields
    >>> Conf  # signature: c(a: Any = 1, b: Any = 'x')
    types.Builds_dict
    >>> instantiate(Conf)  # using Hydra to "instantiate" this build
    {'a': 1, 'b': 'x'}
    >>> instantiate(Conf(a=10, b="hi"))  # overriding configuration values
    {'a': 10, 'b': 'hi'}

    >>> Conf = builds(len, [1, 2, 3])  # specifying positional arguments
    >>> instantiate(Conf)
    3

    Using `builds` to partially-configure a target

    >>> def a_two_tuple(x: int, y: float): return x, y
    >>> PartialConf = builds(a_two_tuple, x=1, zen_partial=True)  # configures only `x`
    >>> partial_func = instantiate(PartialConf)
    >>> partial_func
    functools.partial(<function a_two_tuple at 0x00000220A7820EE0>, x=1)
    >>> partial_func(y=22.0)  # y can be provided after configuration & instantiation
    (1, 22.0)

    Auto-populating parameters:

    >>> Conf = builds(a_two_tuple, populate_full_signature=True)
    >>> # signature: `Builds_a_two_tuple(x: int, y: float)`
    >>> instantiate(Conf(x=1, y=10.0))
    (1, 10.0)

    Inheritance:

    >>> ParentConf = builds(dict, a=1, b=2)
    >>> ChildConf = builds(dict, b=-2, c=-3, builds_bases=(ParentConf,))
    >>> instantiate(ChildConf)
    {'a': 1, 'b': -2, 'c': -3}
    >>> issubclass(ChildConf, ParentConf)
    True

    Leveraging meta-fields for portable, relative interpolation:

    >>> Conf = builds(dict, a="${.s}", b="${.s}", zen_meta=dict(s=-10))
    >>> instantiate(Conf)
    {'a': -10, 'b': -10}
    >>> instantiate(Conf, s=2)
    {'a': 2, 'b': 2}

    Leveraging zen-wrappers to inject unit-conversion capabilities. Let's take
    a function that converts Farenheit to Celcius, and wrap it so that it converts
    to Kelvin instead.

    >>> def faren_to_celsius(temp_f):
    ...     return ((temp_f - 32) * 5) / 9

    >>> def change_celcius_to_kelvin(celc_func):
    ...     def wraps(*args, **kwargs):
    ...         return 273.15 + celc_func(*args, **kwargs)
    ...     return wraps

    >>> AsCelcius = builds(faren_to_celsius)
    >>> AsKelvin = builds(faren_to_celsius, zen_wrappers=change_celcius_to_kelvin)
    >>> instantiate(AsCelcius, temp_f=32)
    0.0
    >>> instantiate(AsKelvin, temp_f=32)
    273.15
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

    del pos_args

    if not callable(target):
        raise TypeError(
            _utils.building_error_prefix(target)
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

    target_field: List[Union[Tuple[str, Type[Any]], Tuple[str, Type[Any], Field[Any]]]]

    if zen_partial or zen_meta or validated_wrappers:
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
                    _PARTIAL_TARGET_FIELD_NAME,
                    bool,
                    _utils.field(default=True, init=False),
                ),
            )

        if zen_meta:
            target_field.append(
                (
                    _META_FIELD_NAME,
                    _utils.sanitized_type(Tuple[str, ...]),
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
                        _utils.sanitized_type(
                            Union[Union[str, Builds], Tuple[Union[str, Builds], ...]]
                        ),
                        _utils.field(default=validated_wrappers[0], init=False),
                    ),
                )
            else:
                target_field.append(
                    (
                        _ZEN_WRAPPERS_FIELD_NAME,
                        _utils.sanitized_type(
                            Union[Union[str, Builds], Tuple[Union[str, Builds], ...]]
                        ),
                        _utils.field(default=validated_wrappers, init=False),
                    ),
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
                    default=tuple(create_just_if_needed(x) for x in _pos_args),
                    init=False,
                ),
            )
        )

    try:
        signature_params = inspect.signature(target).parameters
        target_has_valid_signature: bool = True
    except ValueError:
        if populate_full_signature:
            raise ValueError(
                _utils.building_error_prefix(target)
                + f"{target} does not have an inspectable signature. "
                f"`builds({_utils.safe_name(target)}, populate_full_signature=True)` is not supported"
            )
        signature_params: Mapping[str, inspect.Parameter] = {}
        # We will turn off signature validation for objects that didn't have
        # a valid signature. This will enable us to do things like `build(dict, a=1)`
        target_has_valid_signature: bool = False

    # this properly resolves forward references, whereas the annotations
    # from signature do not
    try:
        if inspect.isclass(target) and hasattr(type, "__init__"):
            # target is class object...
            # calling `get_type_hints(target)` returns empty dict
            type_hints = get_type_hints(target.__init__)
        else:
            type_hints = get_type_hints(target)
    except (TypeError, NameError):
        # TypeError: Covers case for ufuncs, which do not have inspectable type hints
        # NameError: Covers case for unresolved forward reference
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
                    _utils.building_error_prefix(target)
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
                    _utils.building_error_prefix(target)
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
                            _utils.building_error_prefix(target)
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
                    _utils.building_error_prefix(target)
                    + f"{_utils.get_obj_path(target)} takes {_permissible} positional args, but "
                    f"{len(_pos_args)} were specified via `builds`"
                )

    # Create valid dataclass fields from the user-specified values
    #
    # user_specified_params: arg-name -> (arg-name, arg-type, field-w-value)
    #  - arg-type: taken from the parameter's annotation in the target's signature
    #    and is resolved to one of the type-annotations supported by hydra if possible,
    #    otherwise, is Any
    #  - arg-value: mutable values are automatically specified using default-factory
    user_specified_named_params: Dict[str, Tuple[str, type, Field]] = {
        name: (
            name,
            _utils.sanitized_type(type_hints.get(name, Any))
            # OmegaConf's type-checking occurs before instantiation occurs.
            # This means that, e.g., passing `Builds[int]` to a field `x: int`
            # will fail Hydra's type-checking upon instantiation, even though
            # the recursive instantiation will appropriately produce `int` for
            # that field. This will not be addressed by hydra/omegaconf:
            #    https://github.com/facebookresearch/hydra/issues/1759
            # Thus we will auto-broaden the annotation when we see that the user
            # has specified a `Builds` as a default value.
            if not is_builds(value) or hydra_recursive is False else Any,
            sanitized_default_value(value),
        )
        for name, value in kwargs_for_target.items()
    }

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
        user_specified_named_params.update(
            {
                name: (name, Any, sanitized_default_value(value))
                for name, value in zen_meta.items()
            }
        )

    if populate_full_signature is True:
        # Populate dataclass fields based on the target's signature.
        #
        # A user-specified parameter value (via `kwargs_for_target`) takes precedent over
        # the default-value from the signature

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
            elif param.name in fields_set_by_bases:
                # don't populate a parameter that can be derived from a base
                continue
            else:
                # any parameter whose default value is None is automatically
                # annotated with `Optional[...]`. This improves flexibility with
                # Hydra's type-validation
                param_field = (
                    param.name,
                    _utils.sanitized_type(
                        type_hints.get(param.name, Any),
                        wrap_optional=param.default is None,
                    ),
                )

                if param.default is inspect.Parameter.empty:
                    if not zen_partial:
                        # No default value specified in signature or by the user.
                        # We don't include these fields if the user specified a partial build
                        # because we assume that they want to fill these in by using partial
                        base_fields.append(param_field)
                else:
                    param_field += (sanitized_default_value(param.default),)
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

    if dataclass_name is None:
        if zen_partial is False:
            dataclass_name = f"Builds_{_utils.safe_name(target)}"
        else:
            dataclass_name = f"PartialBuilds_{_utils.safe_name(target)}"

    out = make_dataclass(
        dataclass_name, fields=base_fields, bases=builds_bases, frozen=frozen
    )

    if zen_partial is False and hasattr(out, _PARTIAL_TARGET_FIELD_NAME):
        # `out._partial_target_` has been inherited; this will lead to an error when
        # hydra-instantiation occurs, since it will be passed to target.
        # There is not an easy way to delete this, since it comes from a parent class
        raise TypeError(
            _utils.building_error_prefix(target)
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
            # ensures we conver this branch in tests
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
        uses_zen_processing(x) and getattr(x, _PARTIAL_TARGET_FIELD_NAME, False) is True
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
    Returns the target-object from a targeted structured config.

    The target is imported and returned if the config's ``_target_``
    field is a string that indicates its location.

    Parameters
    ----------
    obj : HasTarget | HasPartialTarget

    Returns
    -------
    target : Any

    Examples
    --------
    >>> from hydra_zen import builds, just, get_target, load_from_yaml
    >>> get_target(builds(int))
    int
    >>> get_target(just(str))
    str

    This works even if the ``_target_`` field specifies a string.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class A:
    ...     _target_: str = "builtins.dict"
    >>> get_target(A)
    dict

    This function is useful for accessing a target's type from a config
    without having to instantiate the target. For example, suppose we want
    to access a type from a yaml-serialized config.

    >>> ModelConfig = load_from_yaml("model.yaml")
    >>> get_target(ModelConfig)
    CustomClassifier
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
            f" {_TARGET_FIELD_NAME} or named {_PARTIAL_TARGET_FIELD_NAME} that"
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
        If True, then the resulting config will instantiate as
        ``functools.partial(hydra_target, *pos_args, **kwargs_for_target)`` rather than
        ``hydra_target(*pos_args, **kwargs_for_target)``.

        Thus this enables the partial-configuration of objects.

        Specifying ``zen_partial=True`` and ``populate_full_signature=True`` together will
        populate the dataclass' signature only with parameters that are specified by the
        user or that have default values specified in the target's signature. I.e. it is
        presumed that un-specified parameters are to be excluded from the partial configuration.

    zen_wrappers : None | Callable | Builds | InterpStr | Sequence[None | Callable | Builds | InterpStr]
        One or more wrappers, which will wrap ``hydra_target`` prior to instantiation.
        E.g. specifying the wrappers ``[f1, f2, f3]`` will instantiate as::

            f3(f2(f1(hydra_target)))(*args, **kwargs)

        Wrappers can also be specified as interpolated strings [1]_ or targeted structured
        configs.

    zen_meta : Optional[Mapping[str, Any]]
        Specifies field-names and corresponding values that will be included in the
        resulting dataclass, but that will *not* be used to build ``hydra_target``
        via instantiation. These are called "meta" fields.

    populate_full_signature : bool, optional (default=False)
        If ``True``, then the resulting dataclass's signature and fields will be populated
        according to the signature of ``hydra_target``.

        Values specified in **kwargs_for_target take precedent over the corresponding
        default values from the signature.

        This option is not available for objects with inaccessible signatures, such as
        NumPy's various ufuncs.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how hydra handles the non-primitive objects passed to ``hydra_target`` [3]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with
          the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without
          a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    frozen : bool, optional (default=False)
        If ``True``, the resulting dataclass will create frozen (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance will raise ``FrozenInstanceError``
        at runtime.

    builds_bases : Tuple[DataClass, ...]
        Specifies a tuple of parent classes that the resulting dataclass inherits from.
        A ``PartialBuilds`` class (resulting from ``zen_partial=True``) cannot be a parent
        of a ``Builds`` class (i.e. where `zen_partial=False` was specified).

    returns
    -------
    builds
        The function `builds`, but with customized default-values.

    References
    ----------
    .. [1] https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.

    Examples
    --------
    >>> from hydra_zen import builds, make_custom_builds_fn, instantiate

    The following will create a `builds` function whose default-value
    for ``zen_partial`` has been set to ``True``.

    >>> pbuilds = make_custom_builds_fn(zen_partial=True)

    I.e. using ``pbuilds(...)`` is equivalent to using
    ``builds(..., zen_partial=True)``.

    >>> instantiate(pbuilds(int))  # calls `functools.partial(int)`
    functools.partial(<class 'int'>)
    >>> instantiate(builds(int, zen_partial=True))  # manually-overriding default
    functools.partial(<class 'int'>)

    You can still specify ``zen_partial`` on a per-case basis with ``pbuilds``

    >>> instantiate(pbuilds(int, zen_partial=False))  # calls `int()`
    0

    Suppose that we want to enable runtime type-checking - using beartype -
    whenever our configs are being instantiated; then the following settings
    for `builds` would be handy

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
    >>> conf = build_a_bear(f)
    >>> instantiate(conf, x="a")
    "a"
    >>> instantiate(conf, x="c")
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

    Specifies a field's name and/or type-annotation and/or default-value.
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
    ``default`` will be returned as an instance of ``dataclasses.Field``.
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
    default: Any = NOTHING
    name: Union[str, Type[NOTHING]] = NOTHING

    def __post_init__(self):
        if not isinstance(self.name, str):
            if self.name is not NOTHING:
                raise TypeError(f"`ZenField.name` expects a string, got: {self.name}")

        self.hint = _utils.sanitized_type(self.hint)

        if self.default is not NOTHING:
            self.default = sanitized_default_value(self.default)


def make_config(
    *fields_as_args: Union[str, ZenField],
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
    config_name: str = "Config",
    frozen: bool = False,
    bases: Tuple[Type[DataClass], ...] = (),
    **fields_as_kwargs,
) -> Type[DataClass]:
    """
    Creates a structured config with user-defined fieldnames and, optionally,
    associated default values and/or type-annotations.

    Unlike `builds`, `make_config` is not used to configure a particular target
    object, rather, it can be used to create more general structured configs [1]_.

    Parameters
    ----------
    *fields_as_args : str | ZenField
        The names of the fields to be be included in the config. Or,
        the fields' names and their default values and/or their type
        annotations, expressed via `ZenField` instances.

    **fields_as_kwargs : Any | ZenField
        Like ``fields_as_args``, but fieldname/default-value pairs are
        specified as keyword arguments. `ZenField` can also be used here
        to express a fields type-annotation and/or its default value.

        Named parameters of the forms ``hydra_xx``, ``zen_xx``, and ``_zen_xx`` are reserved
        to ensure future-compatibility, and cannot be specified by the user.

    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this dataclass [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting dataclass.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive objects passed to configuations [3]_.

        - ``"none"``: Passed objects are DictConfig and ListConfig, default
        - ``"partial"``: Passed objects are converted to dict and list, with
            the exception of Structured Configs (and their fields).
        - ``"all"``: Passed objects are dicts, lists and primitives without
            a trace of OmegaConf containers

        If ``None``, the ``_convert_`` attribute is not set on the resulting dataclass.

    bases : Tuple[Type[DataClass], ...], optional (default=())
        Base classes that the resulting config class will inherit from.

    frozen : bool, optional (default=False)
        If ``True``, the resulting config class will produce 'frozen' (i.e. immutable) instances.
        I.e. setting/deleting an attribute of an instance of the config will raise
        ``dataclasses.FrozenInstanceError`` at runtime.

    config_name : str, optional (default="Config")
        The class name of the resulting config class.

    Returns
    -------
    Config : Type[DataClass]
        The resulting config class; a dataclass that possess the user-specified fields.

    Notes
    -----
    Any field specified without a type-annotation is automatically annotated with ``typing.Any``.
    Hydra only supports a narrow subset of types [4]_; `make_config` will automatically 'broaden'
    any user-specified annotations so that they are compatible with Hydra.

    `make_config` will automatically manipulate certain types of default values to ensure that
    they can be utilized in the resulting dataclass and by Hydra:

    - Mutable default values will automatically be packaged in a default factory function [5]_
    - A default value that is a class-object or function-object will automatically be wrapped by
    `just`, to ensure that the resulting config is serializable by Hydra.

    For finer-grain control over how type-annotations and default values are managed, consider using
    ``dataclasses.make_dataclass`` [6]_.

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    just : Create a config that 'just' returns a class-object or function, without instantiating/calling it.

    References
    ----------
    .. [1] https://hydra.cc/docs/next/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#parameter-conversion-strategies
    .. [4] https://hydra.cc/docs/next/tutorials/structured_config/intro/#structured-configs-supports
    .. [5] https://docs.python.org/3/library/dataclasses.html#default-factory-functions
    .. [6] https://docs.python.org/3/library/dataclasses.html#dataclasses.make_dataclass

    Examples
    --------
    >>> from hydra_zen import make_config, to_yaml
    >>> def pp(x): return print(to_yaml(x))  # pretty-print config as yaml

    Let's create a bare-bones config with two fields, named 'a' and 'b'.

    >>> Conf1 = make_config("a", "b")  # sig: `Conf(a: Any, b: Any)`
    >>> pp(Conf1)
    a: ???
    b: ???

    Now we'll configure these fields with particular values:

    >>> pp(Conf1(1, "hi"))
    a: 1
    b: hi

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

    >>> pp(make_config(c=2, bases=(Conf2, Conf1)))
    a: ???
    b: ???
    unit: ???
    data:
    - -10
    - -20
    c: 2

    **Using ZenField to Provide Type Information**

    The `ZenField` class can be used to include a type-annotation in association
    with a field.

    >>> from hydra_zen import ZenField as zf
    >>> ProfileConf = make_config(username=zf(str), age=zf(int))
    >>> # signature: ProfileConf(username: str, age: int)

    Providing type annotations is optional, but doing so enables Hydra to perform
    checks at runtime to ensure that a configured value matches its associated type [4]_.

    >>> pp(ProfileConf(username="piro", age=False))  # age should be an integer
    <ValidationError: Value 'False' could not be converted to Integer>

    These default values can be provides alongside type-annotations

    >>> C = make_config(age=zf(int, 0))  # signature: C(age: int = 0)

    `ZenField` can also be used to specify ``fields_as_args``; here, field names
    must be specified as well.

    >>> C2 = make_config(zf(name="username", hint=str), age=zf(int, 0))
    >>> # signature: C2(username: str, age: int = 0)
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
            normalized_fields[_field.name] = _field

    for name, value in fields_as_kwargs.items():
        if not isinstance(value, ZenField):
            normalized_fields[name] = ZenField(name=name, default=value)
        else:
            normalized_fields[name] = value

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
                    f.hint
                    # f.default: Field
                    # f.default.default: Any
                    if not is_builds(f.default.default) or hydra_recursive is False
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
