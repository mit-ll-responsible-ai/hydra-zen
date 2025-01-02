# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import warnings
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import MISSING, field as _field, is_dataclass
from keyword import iskeyword
from typing import Any, Callable, Final, Optional, TypeVar, Union, cast, overload

from omegaconf import II, DictConfig, ListConfig
from typing_extensions import Literal, TypeGuard

from hydra_zen._compatibility import HYDRA_SUPPORTS_OBJECT_CONVERT
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

COMMON_MODULES_WITH_OBFUSCATED_IMPORTS: tuple[str, ...] = (
    "random",
    "numpy",
    "numpy.random",
    "jax.numpy",
    "jax",
    "torch",
)
UNKNOWN_NAME: Final[str] = "<unknown>"
KNOWN_MUTABLE_TYPES: set[
    Union[type[list[Any]], type[dict[Any, Any]], type[set[Any]]]
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
        return obj.__name__.replace("<lambda>", "lambda")

    if repr_allowed and hasattr(obj, "__repr__"):
        return repr(obj).replace("<lambda>", "lambda")

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


_DATACLASS_OPTION_KEYS: frozenset[str] = (
    DataclassOptions.__required_keys__ | DataclassOptions.__optional_keys__
)

_STRICT_DATACLASS_OPTION_KEYS: frozenset[str] = (
    StrictDataclassOptions.__required_keys__ | StrictDataclassOptions.__optional_keys__
)
_STRICT_DATACLASS_OPTION_KEYS.copy()


def parse_dataclass_options(
    options: Mapping[str, Any], include_module: bool = True
) -> DataclassOptions:
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
            if val is not None and (
                not isinstance(val, str)
                or not all(
                    v.isidentifier() and not iskeyword(v) for v in val.split(".")
                )
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
        elif name == "target":
            if not isinstance(val, str) or not all(
                x.isidentifier() for x in val.split(".")
            ):
                raise TypeError(
                    f"dataclass option `target` must be a string and an import path, "
                    f"got {val!r}"
                )
        elif not isinstance(val, bool):
            raise TypeError(
                f"dataclass option `{name}` must be of type `bool`. Got {val} "
                f"(type: {type(val)})"
            )
        merged[name] = val
    if (
        include_module
        and "module" not in merged
        and "module" in _STRICT_DATACLASS_OPTION_KEYS
    ):  # pragma: no cover
        # For Python 3.12+ we want the default module to
        # remain "types" rather than being inferred as some
        # internal hydra-zen module.
        merged["module"] = "types"
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
