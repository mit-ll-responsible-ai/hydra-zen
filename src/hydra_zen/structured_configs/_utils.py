# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import sys
from dataclasses import is_dataclass
from enum import Enum
from typing import Any, Callable, Tuple, TypeVar, Union

from typing_extensions import Final

from hydra_zen.typing import Importable

COMMON_MODULES_WITH_OBFUSCATED_IMPORTS: Tuple[str, ...] = ("numpy",)
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


def get_obj_path(obj: Importable) -> str:
    name = safe_name(obj, repr_allowed=False)

    if name == UNKNOWN_NAME:
        raise AttributeError(f"{obj} does not have a `__name__` attribute")

    try:
        module = obj.__module__
    except AttributeError as e:
        for module in COMMON_MODULES_WITH_OBFUSCATED_IMPORTS:
            # NumPy's ufuncs do not have an inspectable `__module__` attribute, so we
            # check to see if the object lives in NumPy's top-level namespace.
            #
            # This can be used for other libraries as well
            if hasattr(sys.modules.get(module), obj.__name__):
                break
        else:  # pragma: no cover
            raise e

    return f"{module}.{name}"


def interpolated(func: Union[str, Callable], *literals: Any) -> str:
    """Produces an hydra-style interpolated string for calling the provided
    function on the literals

    Parameters
    ----------
    func : Union[str, Callable]
        The name of the function to use in the interpolation. The name
        will be inferred if a function is provided.

    literals : Any
        Position-only literals to be fed to the function.

    Notes
    -----
    See https://omegaconf.readthedocs.io/en/latest/usage.html#custom-interpolations
    for more details about leveraging custom interpolations in omegaconf/hydra.

    Examples
    --------
    >>> def add(x, y): return x + y
    >>> interpolated(add, 1, 2)
    '${add:1,2}'
    """
    if not isinstance(func, str) and not hasattr(func, "__name__"):  # pragma: no cover
        raise TypeError(
            f"`func` must be a string or have a `__name__` field, got: {func}"
        )
    name = func if isinstance(func, str) else func.__name__
    return f"${{{name}:{','.join(str(i) for i in literals)}}}"


def sanitized_type(type_: type) -> type:
    """Returns ``type_`` unchanged if it is supported as an annotation by hydra,
    otherwise returns ``Any``

    Examples
    --------
    >>> sanitized_type(int)
    int

    >>> sanitized_type(frozenset)  # not supported by hydra
    typing.Any"""
    # TODO: Fully mirror hydra's range of supported type-annotations
    if (
        type_ is Any
        or type_ in HYDRA_SUPPORTED_PRIMITIVES
        or is_dataclass(type_)
        or (isinstance(type_, type) and issubclass(type_, Enum))
    ):
        return type_
    return Any
