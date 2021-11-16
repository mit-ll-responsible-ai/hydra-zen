from typing import Any, Callable, Dict, Set

# `set` support implemented in _implementations.py
ZEN_SUPPORTED_PRIMITIVES: Set[type] = {set}
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}
