import inspect
from functools import wraps
from typing import Callable, Sequence, TypeVar, cast, get_type_hints

from omegaconf import ListConfig

_T = TypeVar("_T", bound=Callable)

__all__ = ["convert_sequences"]


def convert_sequences(obj: _T) -> _T:
    """
    Hydra is only able to read sequences as lists (or ListConfig).
    This wrapper will cast these lists to their desired type, based
    on the annotated-type associated with that sequence.

    This is a no-op for objects that don't have any annotated sequence
    types.

    This is strictly an experimental utility. Use at your own risk.

    Parameters
    ----------
    obj : Callable

    Returns
    -------
    wrapped_obj

    Examples
    --------
    >>> from hydra_zen import builds, instantiate
    >>> def f(x: tuple): return x

    Without wrapping:

    >>> conf_no_wrap = builds(f, x=(1, 2))
    >>> instantiate(conf_no_wrap)
    [1, 2]

    With wrapping:

    >>> conf_wrapped = builds(f, x=(1, 2), zen_wrappers=convert_sequences)
    >>> instantiate(conf_wrapped)
    (1, 2)
    """
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        hints = get_type_hints(obj.__init__)
    else:
        hints = get_type_hints(obj)

    if not hints:
        return cast(_T, obj)

    # We need to keep track of parameters that are annotated with
    # sequence-like annotations
    list_caster_by_pos = {}  # by pos-index in the signature
    list_caster_by_name = {}  # by param-name in the signature

    sig = inspect.signature(obj)
    for n, name in enumerate(sig.parameters):
        if name not in hints:
            continue
        annotation = hints[name]
        if not isinstance(annotation, type):
            # E.g. annotation = Tuple[int, int]
            # Tuple[int, int].__origin__ -> tuple
            caster = getattr(annotation, "__origin__", None)
            if caster is None:
                continue
        else:
            # E.g. annotation = tuple
            caster = annotation

        if (
            not inspect.isabstract(caster)  # E.g. caster = Sequence
            and issubclass(caster, Sequence)
            and not issubclass(caster, str)  # strings don't need to be cast
            and caster is not list  # annotation is list to begin with
        ):
            list_caster_by_pos[n] = caster
            list_caster_by_name[name] = caster
    min_pos = min(list_caster_by_pos) if list_caster_by_pos else 0

    assert len(list_caster_by_name) == len(list_caster_by_pos)

    if not list_caster_by_name:
        # no annotations associated with sequences
        return cast(_T, obj)

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if min_pos < len(args):
            # at least one positional argument needs to be cast
            args = list(args)
            for pos in list_caster_by_pos:
                if pos < len(args) and isinstance(args[pos], (list, ListConfig)):
                    args[pos] = list_caster_by_pos[pos](args[pos])
        if kwargs:
            named = set(list_caster_by_name).intersection(kwargs)
            for name in named:
                # at least one named-arg needs to be cast
                if isinstance(kwargs[name], (list, ListConfig)):
                    kwargs[name] = list_caster_by_name[name](kwargs[name])

        return obj(*args, **kwargs)

    return cast(_T, wrapper)
