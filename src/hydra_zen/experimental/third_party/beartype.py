import inspect
from functools import wraps
from typing import Callable, Sequence, TypeVar, cast, get_type_hints

import beartype as bt
from omegaconf import ListConfig

_T = TypeVar("_T", bound=Callable)

__all__ = ["validates_with_beartype"]


def validates_with_beartype(obj: _T) -> _T:
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        hints = get_type_hints(obj.__init__)
        obj.__init__ = bt.beartype(obj.__init__)
        target = obj
    else:
        hints = get_type_hints(obj)
        target = bt.beartype(obj)

    list_caster_by_pos = {}
    list_caster_by_name = {}

    if hints:
        sig = inspect.signature(obj)
        for n, name in enumerate(sig.parameters):
            if name not in hints:
                continue
            annotation = hints[name]
            if not isinstance(annotation, type):
                caster = getattr(annotation, "__origin__", None)
                if caster is None:
                    continue
            else:
                caster = annotation

            if (
                not inspect.isabstract(caster)
                and issubclass(caster, Sequence)
                and caster is not list
            ):
                list_caster_by_pos[n] = caster
                list_caster_by_name[name] = caster
        min_pos = min(list_caster_by_pos) if list_caster_by_pos else 0

    if not hints or not list_caster_by_name:
        return cast(_T, target)

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if hints and list_caster_by_pos:
            if min_pos < len(args):
                args = list(args)
                for pos in list_caster_by_pos:
                    if pos < len(args) and isinstance(args[pos], (list, ListConfig)):
                        args[pos] = list_caster_by_pos[pos](args[pos])
            named = set(list_caster_by_name).intersection(kwargs)
            if named:
                for name in named:
                    if isinstance(kwargs[name], (list, ListConfig)):
                        kwargs[name] = list_caster_by_name[name](kwargs[name])

        return target(*args, **kwargs)

    return cast(_T, wrapper)
