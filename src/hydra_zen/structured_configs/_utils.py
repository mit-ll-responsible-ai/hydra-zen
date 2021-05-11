# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import sys
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from typing_extensions import Final

try:
    from typing import get_args, get_origin
except ImportError:  # pragma: no cover
    # remove at Python 3.7 end-of-life
    import collections

    def get_origin(obj: Any) -> Union[None, type]:
        """Get the unsubscripted version of a type.

        Parameters
        ----------
        obj : Any

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
        return getattr(obj, "__origin__", None)

    def get_args(obj: Any) -> Union[Tuple[type, ...], Tuple[List[type], type]]:
        """Get type arguments with all substitutions performed.

        Parameters
        ----------
        obj : Any

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
        if hasattr(obj, "__origin__") and hasattr(obj, "__args__"):
            args = obj.__args__
            if (
                get_origin(obj) is collections.abc.Callable
                and args
                and args[0] is not Ellipsis
            ):
                args = (list(args[:-1]), args[-1])
            return args
        return ()


COMMON_MODULES_WITH_OBFUSCATED_IMPORTS: Tuple[str, ...] = (
    "numpy",
    "numpy.random",
    "jax.numpy",
    "jax",
    "torch",
)
UNKNOWN_NAME: Final[str] = "<unknown>"
HYDRA_SUPPORTED_PRIMITIVES: Final = {int, float, bool, str, Enum}
KNOWN_MUTABLE_TYPES = (list, dict, set)

T = TypeVar("T")


def safe_name(obj: Any, repr_allowed=True) -> str:
    """Tries to get a descriptive name for an object. Returns '<unknown>`
    instead of raising - useful for writing descriptive/dafe error messages."""
    if hasattr(obj, "__qualname__"):
        return obj.__qualname__

    if hasattr(obj, "__name__"):
        return obj.__name__

    if repr_allowed and hasattr(obj, "__repr__"):
        return repr(obj)

    return UNKNOWN_NAME


def building_error_prefix(target) -> str:
    return f"Building: {safe_name(target)} ..\n"


def get_obj_path(obj: Any) -> str:
    name = safe_name(obj, repr_allowed=False)

    if name == UNKNOWN_NAME:
        raise AttributeError(f"{obj} does not have a `__name__` attribute")

    module = getattr(obj, "__module__", None)

    if "<" in name or module is None:
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
        # module is None, which is apparently a thing..: numpy.random.rand.__module__ is None

        # don't use qualname for obfuscated paths
        name = obj.__name__
        for new_module in COMMON_MODULES_WITH_OBFUSCATED_IMPORTS:
            if getattr(sys.modules.get(new_module), name, None) is obj:
                module = new_module
                break
        else:  # pragma: no cover
            name = safe_name(obj)
            raise ModuleNotFoundError(f"{name} is not importable")

    return f"{module}.{name}"


NoneType = type(None)


def sanitized_type(
    type_: type, *, primitive_only: bool = False, wrap_optional: bool = False
) -> type:
    """Returns ``type_`` unchanged if it is supported as an annotation by hydra,
    otherwise returns ``Any``.

    Examples
    --------
    >>> sanitized_type(int)
    int

    >>> sanitized_type(frozenset)  # not supported by hydra
    typing.Any

    >>> sanitized_type(int, wrap_optional=True)
    Union[
    >>> sanitized_type(List[int])
    List[int]

    >>> sanitized_type(List[int], primitive_only=True)
    Any

    >>> sanitized_type(Dict[str, frozenset])
    Dict[str, Any]
    """

    # Warning: mutating `type_` will mutate the signature being inspected
    # Even calling deepcopy(`type_`) silently fails to prevent this.
    origin = get_origin(type_)

    if origin is not None:
        if primitive_only:
            return Any

        args = get_args(type_)
        if origin is Union:
            # Hydra only supports Optional[<type>] unions
            if len(args) != 2 or type(None) not in args:
                # isn't Optional[<type>]
                return Any

            optional_type, none_type = args
            if not isinstance(None, none_type):
                optional_type = none_type
            optional_type = sanitized_type(optional_type)

            if optional_type is Any:  # Union[Any, T] is just Any
                return Any
            return Union[optional_type, NoneType]

        if origin is list or origin is List:
            return List[sanitized_type(args[0], primitive_only=True)] if args else type_

        if origin is dict or origin is Dict:
            return (
                Dict[
                    sanitized_type(args[0], primitive_only=True),
                    sanitized_type(args[1], primitive_only=True),
                ]
                if args
                else type_
            )

        if origin is tuple or origin is Tuple:
            # hydra silently supports tuples of homogenous types
            # It has some weird behavior. It treats `Tuple[t1, t2, ...]` as `List[t1]`
            # It isn't clear that we want to perpetrate this on our end..
            # So we deal with inhomogeneous types as e.g. `Tuple[str, int]` -> `Tuple[Any, Any]`.
            #
            # Otherwise we preserve the annotation as accurately as possible
            if not args:
                return Any  # bare Tuple not supported by hydra
            unique_args = set(args)
            has_ellipses = Ellipsis in unique_args

            _unique_type = (
                sanitized_type(args[0], primitive_only=True)
                if len(unique_args) == 1 or (len(unique_args) == 2 and has_ellipses)
                else Any
            )
            if has_ellipses:
                return Tuple[_unique_type, ...]
            else:
                return Tuple[(_unique_type,) * len(args)]

        return Any

    if (
        type_ is Any
        or type_ in HYDRA_SUPPORTED_PRIMITIVES
        or is_dataclass(type_)
        or (isinstance(type_, type) and issubclass(type_, Enum))
    ):
        if wrap_optional and type_ is not Any:  # pragma: no cover
            # normally get_type_hints automatically resolves Optional[...]
            # when None is set as the default, but this has been flaky
            # for some pytorch-lightning classes. So we just do it ourselves...
            # It might be worth removing this later since none of our standard tests
            # cover it.
            type_ = Optional[type_]
        return type_

    # Needed to cover python 3.6 where __origin__ doesn't normalize to type
    if not primitive_only and type_ in {List, Tuple, Dict}:  # pragma: no cover
        if wrap_optional and type_ is not Any:
            type_ = Optional[type_]
        return type_

    return Any
