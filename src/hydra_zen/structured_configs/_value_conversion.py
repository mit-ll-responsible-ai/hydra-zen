# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Callable, Dict, Type, cast

from hydra_zen.typing import Builds

from ._utils import get_obj_path

# Some primitive support implemented in _implementations.py
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}


@dataclass
class ConfigComplex:
    real: Any
    imag: Any
    _target_: str = get_obj_path(complex)


def convert_complex(value: complex) -> Builds[Type[complex]]:
    return cast(Builds[Type[complex]], ConfigComplex(real=value.real, imag=value.imag))


ZEN_VALUE_CONVERSION[complex] = convert_complex


@dataclass
class ConfigPath:
    _args_: Any
    _target_: str = get_obj_path(Path)


def convert_path(value: Path) -> Builds[Type[Path]]:
    return cast(Builds[Type[Path]], ConfigPath(_args_=(str(value),)))


ZEN_VALUE_CONVERSION[Path] = convert_path
ZEN_VALUE_CONVERSION[PosixPath] = convert_path
ZEN_VALUE_CONVERSION[WindowsPath] = convert_path
