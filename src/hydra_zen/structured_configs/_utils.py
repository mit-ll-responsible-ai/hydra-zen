# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import sys
import warnings
from dataclasses import MISSING, InitVar, field as _field, is_dataclass
from enum import Enum
from keyword import iskeyword
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
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
    get_args,
    get_origin,
    overload,
)

from omegaconf import II, DictConfig, ListConfig
from typing_extensions import (
    Annotated,
    Final,
    Literal,
    ParamSpecArgs,
    ParamSpecKwargs,
    TypeGuard,
    Unpack,
    _AnnotatedAlias,
)

from hydra_zen._compatibility import (
    HYDRA_SUPPORTED_PRIMITIVE_TYPES,
    HYDRA_SUPPORTS_OBJECT_CONVERT,
    OMEGACONF_VERSION,
    Version,
)
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing import DataclassOptions, ZenConvert
from hydra_zen.typing._implementations import (
    DEFAULT_DATACLASS_OPTIONS,
    UNSUPPORTED_DATACLASS_OPTIONS,
    AllConvert,
    Field,
    InterpStr,
    StrictDataclassOptions,
    convert_types,
)

COMMON_MODULES_WITH_OBFUSCATED_IMPORTS: Tuple[str, ...] = (
    "random",
    "numpy",
    "numpy.random",
    "jax.numpy",
    "jax",
    "torch",
)
UNKNOWN_NAME: Final[str] = "<unknown>"
KNOWN_MUTABLE_TYPES: Set[
    Union[Type[List[Any]], Type[Dict[Any, Any]], Type[Set[Any]]]
] = {list, dict, set}

T = TypeVar("T")


# The typeshed definition of `field` has an inaccurate annotation:
#  https://github.com/python/typeshed/blob/b9e1d7d522fe90b98e07d43a764bbe60216bc2c4/stdlib/dataclasses.pyi#L109
# This makes it impossible for `make_dataclass` to by type-correct in the eyes of
# static checkers. See https://github.com/microsoft/pyright/issues/1680 for discussion.
#
# We happen to make rather heavy use of `make_dataclass`, thus we..*sigh*.. we provide
# our own overloads for `field`.
@overload  # `default` and `default_factory` are optional and mutually exclusive.
def field(
    *,
    default: Any,
    init: bool = ...,
    repr: bool = ...,
    hash: Optional[bool] = ...,
    compare: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
) -> Field[Any]:  # pragma: no cover
    ...


@overload
def field(
    *,
    default_factory: Callable[[], Any],
    init: bool = ...,
    repr: bool = ...,
    hash: Optional[bool] = ...,
    compare: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
) -> Field[Any]:  # pragma: no cover
    ...


def field(
    *,
    default: Any = MISSING,
    default_factory: Union[Callable[[], Any], Any] = MISSING,
    init: bool = True,
    repr: bool = True,
    hash: Optional[bool] = None,
    compare: bool = True,
    metadata: Optional[Mapping[Any, Any]] = None,
) -> Field[Any]:
    if default is MISSING:
        return cast(
            Field[Any],
            _field(
                default_factory=default_factory,
                init=init,
                repr=repr,
                hash=hash,
                compare=compare,
                metadata=metadata,
            ),
        )
    else:
        return cast(
            Field[Any],
            _field(
                default=default,
                init=init,
                repr=repr,
                hash=hash,
                compare=compare,
                metadata=metadata,
            ),
        )


def safe_name(obj: Any, repr_allowed: bool = True) -> str:
    """Tries to get a descriptive name for an object. Returns '<unknown>`
    instead of raising - useful for writing descriptive/dafe error messages."""

    if hasattr(obj, "__name__"):
        return obj.__name__

    if repr_allowed and hasattr(obj, "__repr__"):
        return repr(obj)

    return UNKNOWN_NAME


def is_classmethod(obj: Any) -> bool:
    """
    https://stackoverflow.com/a/19228282/6592114

    Credit to: Martijn Pieters
    License: CC BY-SA 4.0 (free to copy/redistribute/remix/transform)"""

    if not inspect.ismethod(obj):
        return False

    bound_to = getattr(obj, "__self__", None)
    if not isinstance(bound_to, type):
        # must be bound to a class
        return False
    name = safe_name(obj)

    if name == UNKNOWN_NAME:  # pragma: no cover
        return False

    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False  # pragma: no cover


def building_error_prefix(target: Any) -> str:
    return f"Building: {safe_name(target)} ..\n"


NoneType = type(None)
_supported_types = HYDRA_SUPPORTED_PRIMITIVE_TYPES | {
    list,
    dict,
    tuple,
    List,
    Tuple,
    Dict,
}


def sanitized_type(
    type_: Any,
    *,
    primitive_only: bool = False,
    wrap_optional: bool = False,
    nested: bool = False,
) -> type:
    """Returns ``type_`` unchanged if it is supported as an annotation by hydra,
    otherwise returns ``Any``.

    Examples
    --------
    >>> sanitized_type(int)
    <class 'int'>

    >>> sanitized_type(frozenset)  # not supported by hydra
    typing.Any

    >>> sanitized_type(int, wrap_optional=True)
    typing.Union[int, NoneType]

    >>> sanitized_type(List[int])
    List[int]

    >>> sanitized_type(List[int], primitive_only=True)
    Any

    >>> sanitized_type(Dict[str, frozenset])
    Dict[str, Any]
    """
    if hasattr(type_, "__supertype__"):
        # is NewType
        return sanitized_type(
            type_.__supertype__,
            primitive_only=primitive_only,
            wrap_optional=wrap_optional,
            nested=nested,
        )

    if OMEGACONF_VERSION < Version(2, 2, 3):  # pragma: no cover
        try:
            type_ = {list: List, tuple: Tuple, dict: Dict}.get(type_, type_)
        except TypeError:
            pass

    # Warning: mutating `type_` will mutate the signature being inspected
    # Even calling deepcopy(`type_`) silently fails to prevent this.
    origin = get_origin(type_)

    if origin is not None:
        # Support for Annotated[x, y]
        # Python 3.9+
        # # type_: Annotated[x, y]; origin -> Annotated; args -> (x, y)
        if origin is Annotated:  # pragma: no cover
            return sanitized_type(
                get_args(type_)[0],
                primitive_only=primitive_only,
                wrap_optional=wrap_optional,
                nested=nested,
            )

        # Python 3.7-3.8
        # type_: Annotated[x, y]; origin -> x
        if isinstance(type_, _AnnotatedAlias):  # pragma: no cover
            return sanitized_type(
                origin,
                primitive_only=primitive_only,
                wrap_optional=wrap_optional,
                nested=nested,
            )

        if primitive_only:  # pragma: no cover
            return Any

        args = get_args(type_)
        if origin is Union:
            # Hydra only supports Optional[<type>] unions
            if len(args) != 2 or NoneType not in args:
                # isn't Optional[<type>]
                return Any

            args = cast(Tuple[type, type], args)

            optional_type, none_type = args
            if none_type is not NoneType:
                optional_type = none_type

            optional_type = sanitized_type(optional_type)

            if optional_type is Any:  # Union[Any, T] is just Any
                return Any

            return Union[optional_type, NoneType]

        if origin is list or origin is List:
            if args:
                return List[sanitized_type(args[0], primitive_only=False, nested=True)]
            return List

        if origin is dict or origin is Dict:
            if args:
                KeyType = sanitized_type(args[0], primitive_only=True, nested=True)
                ValueType = sanitized_type(args[1], primitive_only=False, nested=True)
                return Dict[KeyType, ValueType]
            return Dict

        if (origin is tuple or origin is Tuple) and not nested:
            # hydra silently supports tuples of homogeneous types
            # It has some weird behavior. It treats `Tuple[t1, t2, ...]` as `List[t1]`
            # It isn't clear that we want to perpetrate this on our end..
            # So we deal with inhomogeneous types as e.g. `Tuple[str, int]` -> `Tuple[Any, Any]`.
            #
            # Otherwise we preserve the annotation as accurately as possible
            if not args:
                return Any if OMEGACONF_VERSION < (2, 2, 3) else Tuple

            args = cast(Tuple[type, ...], args)
            unique_args = set(args)

            if any(get_origin(tp) is Unpack for tp in unique_args):
                # E.g. Tuple[*Ts]
                return Tuple[Any, ...]

            has_ellipses = Ellipsis in unique_args

            # E.g. Tuple[int, int, int] or Tuple[int, ...]
            _unique_type = (
                sanitized_type(args[0], primitive_only=False, nested=True)
                if len(unique_args) == 1 or (len(unique_args) == 2 and has_ellipses)
                else Any
            )

            if has_ellipses:
                return Tuple[_unique_type, ...]
            else:
                return Tuple[(_unique_type,) * len(args)]  # type: ignore

        return Any

    if isinstance(type_, type) and issubclass(type_, Path):
        type_ = Path

    if isinstance(type_, (ParamSpecArgs, ParamSpecKwargs)):  # pragma: no cover
        # Python 3.7 - 3.9
        # these aren't hashable -- can't check for membership in set
        return Any

    if isinstance(type_, InitVar):
        return sanitized_type(
            type_.type,
            primitive_only=primitive_only,
            wrap_optional=wrap_optional,
            nested=nested,
        )
    if (
        type_ is Any
        or type_ in _supported_types
        or is_dataclass(type_)
        or (isinstance(type_, type) and issubclass(type_, Enum))
    ):
        if sys.version_info[:2] == (3, 6) and type_ is Dict:  # pragma: no cover
            type_ = Dict[Any, Any]

        if wrap_optional and type_ is not Any:  # pragma: no cover
            # normally get_type_hints automatically resolves Optional[...]
            # when None is set as the default, but this has been flaky
            # for some pytorch-lightning classes. So we just do it ourselves...
            # It might be worth removing this later since none of our standard tests
            # cover it.
            type_ = Optional[type_]
        return type_

    return Any


def is_interpolated_string(x: Any) -> TypeGuard[InterpStr]:
    # This is only a necessary check – not a sufficient one – that `x`
    # is a valid interpolated string. We do not verify that it rigorously
    # satisfies omegaconf's grammar
    return isinstance(x, str) and len(x) > 3 and x.startswith("${") and x.endswith("}")


def check_suspicious_interpolations(
    validated_wrappers: Sequence[Any], zen_meta: Mapping[str, Any], target: Any
):
    """Looks for patterns among zen_meta fields and interpolated fields in
    wrappers. Relative interpolations pointing to the wrong level will produce
    a warning"""
    for _w in validated_wrappers:
        if is_interpolated_string(_w):
            _lvl = _w.count(".")  # level of relative-interp
            _field_name = _w.replace(".", "")[2:-1]
            if (
                _lvl
                and _field_name in zen_meta
                and _lvl != (1 if len(validated_wrappers) == 1 else 2)
            ):
                _expected = II(
                    "." * (1 if len(validated_wrappers) == 1 else 2) + _field_name
                )

                warnings.warn(
                    building_error_prefix(target)
                    + f"A zen-wrapper is specified via the interpolated field, {_w},"
                    f" along with the meta-field name {_field_name}, however it "
                    f"appears to point to the wrong level. It is likely you should "
                    f"change {_w} to {_expected}"
                )
                yield _expected


def valid_defaults_list(hydra_defaults: Any) -> bool:
    """
    Raises
    ------
    HydraZenValidationError: Duplicate _self_ entries"""
    if not isinstance(hydra_defaults, (list, ListConfig)):
        return False

    has_self = False
    for item in hydra_defaults:
        if item == "_self_":
            if not has_self:
                has_self = True
                continue
            raise HydraZenValidationError(
                "`hydra_defaults` cannot have more than one '_self_' entry"
            )

        if isinstance(item, (dict, DictConfig)):
            for k, v in item.items():
                if not isinstance(k, str):
                    return False

                if (
                    not isinstance(v, (str, list, ListConfig))
                    and v is not None
                    and v != MISSING
                ):
                    return False
        elif isinstance(item, str):
            continue
        elif is_dataclass(item):
            # no validation here
            continue
        else:
            return False

    if not has_self:
        warnings.warn(
            "Defaults list is missing `_self_`. See https://hydra.cc/docs/upgrades/1.0_to_1.1/default_composition_order for more information",
            category=UserWarning,
        )
    return True


def merge_settings(
    user_settings: Optional[ZenConvert], default_settings: AllConvert
) -> AllConvert:
    """Merges settings as `default_settings.update(user_settings)`"""
    if user_settings is not None and not isinstance(user_settings, Mapping):
        raise TypeError(
            f"`zen_convert` must be None or Mapping[str, Any] (e.g. dict). Got {user_settings}"
        )
    settings = default_settings.copy()
    if user_settings:
        for k, v in user_settings.items():
            if k not in convert_types:
                raise ValueError(
                    f"The key `{k}` is not a valid zen_convert setting. The available settings are: {', '.join(sorted(convert_types))}"
                )
            if not isinstance(v, convert_types[k]):
                raise TypeError(
                    f"Setting {k}={v} specified a value of the wrong type. Expected type: {convert_types[k].__name__}"
                )
            settings[k] = v
    return settings


_DATACLASS_OPTION_KEYS: FrozenSet[str] = (
    DataclassOptions.__required_keys__ | DataclassOptions.__optional_keys__
)

_STRICT_DATACLASS_OPTION_KEYS: FrozenSet[str] = (
    StrictDataclassOptions.__required_keys__ | StrictDataclassOptions.__optional_keys__
)
_STRICT_DATACLASS_OPTION_KEYS.copy()


def parse_dataclass_options(options: Mapping[str, Any]) -> DataclassOptions:
    """
    Ensures `options` adheres to `DataclassOptions` and merges hydra-zen defaults
    for missing options.

    All valid `@dataclass`/`make_dataclass` options are supported, even for features
    introduced in later versions of Python. This function will remove valid options
    that are not supported for by the current Python version.

    Parameters
    ----------
    options : Mapping[str, Any]
        User-specified options for `zen_dataclass` to be validated.

    Returns
    -------
    DataclassOptions

    Examples
    --------
    >>> parse_dataclass_options({})
    {'unsafe_hash': True}

    >>> parse_dataclass_options({"unsafe_hash": False, "cls_name": "Foo"})
    {'unsafe_hash': False, 'cls_name': 'Foo'}

    >>> parse_dataclass_options({"moo": 1})
    ValueError: moo is not a valid dataclass option.

    Options that are supported by `make_dataclass` for later versions of
    Python are ignored/removed automatically by this function. E.g. the following
    Python 3.10+ option has the following behavior in Python 3.9:

    >>> parse_dataclass_options({"slots": False})
    {'unsafe_hash': True}
    """
    if not isinstance(options, Mapping):
        raise ValueError(
            f"`zen_dataclass_options` is expected to be `None` or dict[str, bool]. Got "
            f"{options} (type: {type(options)})."
        )

    merged = DEFAULT_DATACLASS_OPTIONS.copy()

    for name, val in options.items():
        if name in UNSUPPORTED_DATACLASS_OPTIONS:
            continue
        elif name not in _DATACLASS_OPTION_KEYS:
            raise ValueError(f"{name} is not a valid dataclass option.")

        if name == "module":
            if not isinstance(val, str) or not all(
                v.isidentifier() and not iskeyword(v) for v in val.split(".")
            ):
                raise ValueError(
                    f"dataclass option `{name}` must be a valid module name, got {val}"
                )
        elif name == "cls_name":
            if val is not None and (not isinstance(val, str) or not val.isidentifier()):
                raise ValueError(
                    f"dataclass option `{name}` must be a valid identifier, got {val}"
                )
        elif name == "bases":
            if not isinstance(val, Iterable) or any(
                not (is_dataclass(_b) and isinstance(_b, type)) for _b in val
            ):
                raise TypeError(
                    f"dataclass option `{name}` must be a tuple of dataclass types"
                )
        elif name == "namespace":
            if not isinstance(val, Mapping) or any(
                not isinstance(v, str) or not v.isidentifier() for v in val
            ):
                raise ValueError(
                    f"dataclass option `{name}` must be a mapping with string-valued keys "
                    f"that are valid identifiers. Got {val}."
                )
        elif not isinstance(val, bool):
            raise TypeError(
                f"dataclass option `{name}` must be of type `bool`. Got {val} "
                f"(type: {type(val)})"
            )
        merged[name] = val
    return merged


def parse_strict_dataclass_options(
    options: Mapping[str, Any]
) -> TypeGuard[StrictDataclassOptions]:
    return (
        options.keys() <= _STRICT_DATACLASS_OPTION_KEYS
        and StrictDataclassOptions.__required_keys__ <= options.keys()
    )


_HYDRA_CONVERT_OPTIONS = (
    {"none", "partial", "all", "object"}
    if HYDRA_SUPPORTS_OBJECT_CONVERT
    else {"none", "partial", "all"}
)


def validate_hydra_options(
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
) -> None:
    if hydra_recursive is not None and not isinstance(hydra_recursive, bool):
        raise TypeError(
            f"`hydra_recursive` must be a boolean type, got {hydra_recursive}"
        )

    if hydra_convert is not None and hydra_convert not in _HYDRA_CONVERT_OPTIONS:
        raise ValueError(
            f"`hydra_convert` must be 'none', 'partial',"
            f"{' object' if HYDRA_SUPPORTS_OBJECT_CONVERT else ''} or 'all', got: "
            f"{hydra_convert}"
        )
