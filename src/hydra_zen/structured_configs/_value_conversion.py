# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field
from pathlib import Path, PosixPath, WindowsPath
from typing import Any, Callable, Dict, Tuple, Type, cast

from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.typing import Builds

from ._utils import get_obj_path

# Some primitive support implemented in _implementations.py

ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}


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
