# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import sys
import warnings
from dataclasses import MISSING, field as _field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
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
    HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES,
    HYDRA_SUPPORTS_PARTIAL,
    OMEGACONF_VERSION,
    PATCH_OMEGACONF_830,
    Version,
)
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.typing._implementations import (
    AllConvert,
    DataClass_,
    Field,
    InterpStr,
    ZenConvert,
    convert_types,
)

try:
    from typing import get_args, get_origin
except ImportError:  # pragma: no cover
    # remove at Python 3.7 end-of-life
    from collections.abc import Callable as _Callable

    def get_origin(tp: Any) -> Union[None, type]:
        """Get the unsubscripted version of a type.

        Parameters
        ----------
        tp : Any

        Returns
        -------
        Union[None, type]
            Return None for unsupported types.

        Notes
        -----
        Bare `Generic` not supported by this hacked version of `get_origin`

        Examples
        --------
        >>> assert get_origin(Literal[42]) is Literal
        >>> assert get_origin(int) is None
        >>> assert get_origin(ClassVar[int]) is ClassVar
        >>> assert get_origin(Generic[T]) is Generic
        >>> assert get_origin(Union[T, int]) is Union
        >>> assert get_origin(List[Tuple[T, T]][int]) == list
        """
        return getattr(tp, "__origin__", None)

    def get_args(tp: Any) -> Union[Tuple[type, ...], Tuple[List[type], type]]:
        """Get type arguments with all substitutions performed.

        Parameters
        ----------
        tp : Any

        Returns
        -------
        Union[Tuple[type, ...], Tuple[List[type], type]]
            Callable[[t1, ...], r] -> ([t1, ...], r)

        Examples
        --------
        >>> assert get_args(Dict[str, int]) == (str, int)
        >>> assert get_args(int) == ()
        >>> assert get_args(Union[int, Union[T, int], str][int]) == (int, str)
        >>> assert get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
        >>> assert get_args(Callable[[], T][int]) == ([], int)
        """
        if hasattr(tp, "__origin__") and hasattr(tp, "__args__"):
            args = tp.__args__
            if get_origin(tp) is _Callable and args and args[0] is not Ellipsis:
                args = (list(args[:-1]), args[-1])
            return args
        return ()


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


def get_obj_path(obj: Any) -> str:
    name = safe_name(obj, repr_allowed=False)

    if name == UNKNOWN_NAME:
        raise AttributeError(f"{obj} does not have a `__name__` attribute")

    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)

    if (qualname is not None and "<" in qualname) or module is None:
        # NumPy's ufuncs do not have an inspectable `__module__` attribute, so we
        # check to see if the object lives in NumPy's top-level namespace.
        #
        # or..
        #
        # Qualname produced a name from a local namespace.
        # E.g. jax.numpy.add.__qualname__ is '_maybe_bool_binop.<locals>.fn'
        # Thus we defer to the name of the object and look for it in the
        # top-level namespace of the known suspects
        #
        # or...
        #
        # module is None, which is apparently a thing..:
        # __module__ is None for both numpy.random.rand and random.random
        #

        # don't use qualname for obfuscated paths
        for new_module in COMMON_MODULES_WITH_OBFUSCATED_IMPORTS:
            if getattr(sys.modules.get(new_module), name, None) is obj:
                module = new_module
                break
        else:
            raise ModuleNotFoundError(f"{name} is not importable")

    if not is_classmethod(obj):
        return f"{module}.{name}"
    else:

        # __qualname__ reflects name of class that originaly defines classmethod.
        # Does not point to child in case of inheritance.
        #
        # obj.__self__ -> parent object
        # obj.__name__ -> name of classmethod
        return f"{get_obj_path(obj.__self__)}.{obj.__name__}"


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

    no_nested_container = not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES

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
        if isinstance(type_, _AnnotatedAlias):
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
            optional_type: Optional[Any]
            optional_type = sanitized_type(optional_type)

            if optional_type is Any:  # Union[Any, T] is just Any
                return Any
            return Union[optional_type, NoneType]  # type: ignore

        if origin is list or origin is List:
            if args:
                return List[sanitized_type(args[0], primitive_only=no_nested_container, nested=True)]  # type: ignore
            return List

        if origin is dict or origin is Dict:
            if args:
                KeyType = sanitized_type(args[0], primitive_only=True, nested=True)
                ValueType = sanitized_type(
                    args[1], primitive_only=no_nested_container, nested=True
                )
                return Dict[KeyType, ValueType]  # type: ignore
            return Dict

        if (origin is tuple or origin is Tuple) and not nested:
            # hydra silently supports tuples of homogenous types
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
                sanitized_type(args[0], primitive_only=no_nested_container, nested=True)
                if len(unique_args) == 1 or (len(unique_args) == 2 and has_ellipses)
                else Any
            )

            if has_ellipses:
                return Tuple[_unique_type, ...]  # type: ignore
            else:
                return Tuple[(_unique_type,) * len(args)]  # type: ignore

        return Any

    if HYDRA_SUPPORTS_PARTIAL and isinstance(type_, type) and issubclass(type_, Path):
        type_ = Path

    if isinstance(type_, (ParamSpecArgs, ParamSpecKwargs)):
        # these aren't hashable -- can't check for membership in set
        return Any

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
            type_ = Optional[type_]  # type: ignore
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


def mutable_default_permitted(bases: Iterable[DataClass_], field_name: str) -> bool:
    if not PATCH_OMEGACONF_830:
        return True
    else:  # pragma: no cover
        for base in bases:
            if (
                field_name in base.__dataclass_fields__
                and base.__dataclass_fields__[field_name].default is not MISSING
            ):
                # see https://github.com/omry/omegaconf/issues/830
                return False
        return True


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


def validate_hydra_options(
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
) -> None:
    if hydra_recursive is not None and not isinstance(hydra_recursive, bool):
        raise TypeError(
            f"`hydra_recursive` must be a boolean type, got {hydra_recursive}"
        )

    if hydra_convert is not None and hydra_convert not in {"none", "partial", "all"}:
        raise ValueError(
            f"`hydra_convert` must be 'none', 'partial', or 'all', got: {hydra_convert}"
        )
