from dataclasses import dataclass
from typing import Any, Callable, Dict, Set, Type, cast

from hydra_zen.typing import Builds

from ._utils import get_obj_path

# `set` support implemented in _implementations.py
ZEN_SUPPORTED_PRIMITIVES: Set[type] = {set, complex}
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}


@dataclass
class _ConfigComplex:
    real: Any
    imag: Any
    _target_: str = get_obj_path(complex)


def _convert_complex(value: complex) -> Builds[Type[complex]]:
    return cast(Builds[Type[complex]], _ConfigComplex(real=value.real, imag=value.imag))


ZEN_VALUE_CONVERSION[complex] = _convert_complex
