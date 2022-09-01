# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Callable, Tuple, Type, TypeVar, cast

from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.typing import Builds, Partial, PartialBuilds

from ._implementations import ZEN_VALUE_CONVERSION, builds
from ._utils import get_obj_path

_T = TypeVar("_T")


@dataclass
class ConfigComplex:
    real: Any
    imag: Any
    _target_: str = field(default=get_obj_path(complex), init=False)


def convert_complex(value: complex) -> Builds[Type[complex]]:
    return cast(Builds[Type[complex]], ConfigComplex(real=value.real, imag=value.imag))


ZEN_VALUE_CONVERSION[complex] = convert_complex


@dataclass
class ConfigPath:
    _args_: Tuple[str]
    _target_: str = field(default=get_obj_path(Path), init=False)


if Path in ZEN_SUPPORTED_PRIMITIVES:  # pragma no cover

    def convert_path(value: Path) -> Builds[Type[Path]]:
        return cast(Builds[Type[Path]], ConfigPath(_args_=(str(value),)))

    ZEN_VALUE_CONVERSION[Path] = convert_path
    ZEN_VALUE_CONVERSION[PosixPath] = convert_path
    ZEN_VALUE_CONVERSION[WindowsPath] = convert_path


# registering value-conversions that depend on `builds`
def _cast_via_tuple(dest_type: Type[_T]) -> Callable[[_T], Builds[Type[_T]]]:
    def converter(value):
        return builds(dest_type, tuple(value))()

    return converter


def _unpack_partial(value: Partial[_T]) -> PartialBuilds[Type[_T]]:
    target = cast(Type[_T], value.func)
    return builds(target, *value.args, **value.keywords, zen_partial=True)()


ZEN_VALUE_CONVERSION[set] = _cast_via_tuple(set)
ZEN_VALUE_CONVERSION[frozenset] = _cast_via_tuple(frozenset)
ZEN_VALUE_CONVERSION[deque] = _cast_via_tuple(deque)

if bytes in ZEN_SUPPORTED_PRIMITIVES:  # pragma: no cover
    ZEN_VALUE_CONVERSION[bytes] = _cast_via_tuple(bytes)

ZEN_VALUE_CONVERSION[bytearray] = _cast_via_tuple(bytearray)
ZEN_VALUE_CONVERSION[range] = lambda value: builds(
    range, value.start, value.stop, value.step
)()
ZEN_VALUE_CONVERSION[Counter] = lambda counter: builds(Counter, dict(counter))()
ZEN_VALUE_CONVERSION[functools.partial] = _unpack_partial
