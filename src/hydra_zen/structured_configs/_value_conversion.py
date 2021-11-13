from typing import Any, Callable, Dict, Tuple

ZEN_SUPPORTED_PRIMITIVES: Tuple[type, ...] = ()
ZEN_VALUE_CONVERSION: Dict[type, Callable[[Any], Any]] = {}
