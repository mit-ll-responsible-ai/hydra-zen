import inspect
from typing import Callable, TypeVar, cast

import beartype as bt

from hydra_zen.experimental.coerce import coerce_sequences

_T = TypeVar("_T", bound=Callable)

__all__ = ["validates_with_beartype"]


def validates_with_beartype(obj: _T) -> _T:
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        obj.__init__ = bt.beartype(obj.__init__)
        target = obj
    else:
        target = bt.beartype(obj)
    target = coerce_sequences(target)
    return cast(_T, target)
