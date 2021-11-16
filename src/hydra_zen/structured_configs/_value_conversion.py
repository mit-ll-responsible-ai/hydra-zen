from typing import Any, Callable, Dict, Tuple

# `set` support implemented in _implementations.py
ZEN_SUPPORTED_PRIMITIVES: Tuple[type, ...] = (set,)
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}
