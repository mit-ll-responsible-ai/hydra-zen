# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
from collections import Counter, deque
from dataclasses import InitVar, dataclass, field
from functools import partial
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Tuple, Type, TypeVar, cast

from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.typing import Builds, Partial, PartialBuilds

from ._implementations import ZEN_VALUE_CONVERSION, builds, sanitized_default_value
from ._utils import get_obj_path

_T = TypeVar("_T")


@dataclass(unsafe_hash=True)
class ConfigComplex:
    real: Any
    imag: Any
    _target_: str = field(default=get_obj_path(complex), init=False)


def convert_complex(value: complex) -> Builds[Type[complex]]:
    return cast(Builds[Type[complex]], ConfigComplex(real=value.real, imag=value.imag))


ZEN_VALUE_CONVERSION[complex] = convert_complex


@dataclass(unsafe_hash=True)
class ConfigPath:
    _args_: Tuple[str]
    _target_: str = field(default=get_obj_path(Path), init=False)


if Path in ZEN_SUPPORTED_PRIMITIVES:  # pragma: no cover

    def convert_path(value: Path) -> Builds[Type[Path]]:
        return cast(Builds[Type[Path]], ConfigPath(_args_=(str(value),)))  # type: ignore

    ZEN_VALUE_CONVERSION[Path] = convert_path
    ZEN_VALUE_CONVERSION[PosixPath] = convert_path
    ZEN_VALUE_CONVERSION[WindowsPath] = convert_path


def _unpack_partial(value: Partial[_T]) -> PartialBuilds[Type[_T]]:
    target = cast(Type[_T], value.func)
    return builds(target, *value.args, **value.keywords, zen_partial=True)()


@dataclass(unsafe_hash=True)
class ConfigFromTuple:
    _args_: Tuple[Any, ...]
    _target_: str

    def __post_init__(self):
        self._args_ = (
            sanitized_default_value(
                tuple(self._args_),
                convert_dataclass=True,
                allow_zen_conversion=True,
                structured_conf_permitted=True,
            ),
        )


@dataclass(unsafe_hash=True)
class ConfigFromDict:
    _args_: Any
    _target_: str

    def __post_init__(self):
        self._args_ = (
            sanitized_default_value(
                dict(self._args_),
                convert_dataclass=True,
                allow_zen_conversion=True,
                structured_conf_permitted=True,
            ),
        )


@dataclass(unsafe_hash=True)
class ConfigRange:
    start: InitVar[int]
    stop: InitVar[int]
    step: InitVar[int]
    _target_: str = field(default=get_obj_path(range), init=False)
    _args_: Tuple[int, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self, start, stop, step):
        self._args_ = (start, stop, step)


ZEN_VALUE_CONVERSION[set] = partial(ConfigFromTuple, _target_=get_obj_path(set))
ZEN_VALUE_CONVERSION[frozenset] = partial(
    ConfigFromTuple, _target_=get_obj_path(frozenset)
)
ZEN_VALUE_CONVERSION[deque] = partial(ConfigFromTuple, _target_=get_obj_path(deque))

if bytes in ZEN_SUPPORTED_PRIMITIVES:  # pragma: no cover
    ZEN_VALUE_CONVERSION[bytes] = partial(ConfigFromTuple, _target_=get_obj_path(bytes))

ZEN_VALUE_CONVERSION[bytearray] = partial(
    ConfigFromTuple, _target_=get_obj_path(bytearray)
)
ZEN_VALUE_CONVERSION[range] = lambda value: ConfigRange(
    value.start, value.stop, value.step
)
ZEN_VALUE_CONVERSION[Counter] = partial(ConfigFromDict, _target_=get_obj_path(Counter))
ZEN_VALUE_CONVERSION[functools.partial] = _unpack_partial
