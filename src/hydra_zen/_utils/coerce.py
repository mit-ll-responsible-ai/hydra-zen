import sys

if sys.version_info < (3, 7):  # pragma: no cover
    raise NotImplementedError(
        "Features that utilize `hydra_zen._utils.coerce` "
        "(e.g. beartype-validation) require Python 3.7 or greater."
    )

import inspect
from collections import deque
from functools import wraps
from typing import (
    Callable,
    NamedTuple,
    Sequence,
    Type,
    TypeVar,
    Union,
    cast,
    get_type_hints,
)

from omegaconf import ListConfig
from typing_extensions import TypeGuard

from hydra_zen.structured_configs._utils import get_args, get_origin

_T = TypeVar("_T", bound=Callable)

__all__ = ["coerce_sequences"]


def _is_namedtuple_type(x) -> TypeGuard[Type[NamedTuple]]:  # pragma: no cover

    try:
        bases = x.__bases__
        fields = x._fields
    except AttributeError:
        return False

    if bases is None:
        return False

    if len(bases) != 1 or bases[0] is not tuple:
        return False

    if not isinstance(fields, tuple):
        return False

    return all(isinstance(fieldname, str) for fieldname in fields)


_NoneType = type(None)

_POSITIONAL_ONLY = inspect.Parameter.POSITIONAL_ONLY
_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_KEYWORD_ONLY = inspect.Parameter.KEYWORD_ONLY
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD


def coerce_sequences(obj: _T) -> _T:
    """
    Hydra is only able to read non-string sequences as lists (or ListConfig).
    This wrapper will cast these lists to their desired type, based
    on the annotated-type associated with that sequence.

    This is a no-op for objects that don't have any annotated non-string
    sequence types.

    This is strictly an experimental utility. Use at your own risk.

    Parameters
    ----------
    obj : Callable

    Returns
    -------
    wrapped_obj

    Notes
    -----
    The supported non-string sequence types are:
    - tuples
    - deques
    - lists
    - named-tuples

    The only Union-based types that are supported are of the form
    ``Optional[<sequence-type>]``.

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
    for n, (name, param) in enumerate(sig.parameters.items()):
        if name not in hints or param.kind in {_VAR_KEYWORD, _VAR_POSITIONAL}:
            continue

        annotation = hints[name]

        _origin = get_origin(annotation)

        if _origin is not None and _origin is Union:
            # Check for Optional[A]
            _args = get_args(annotation)
            if len(_args) == 2:
                if _args[0] is _NoneType:
                    annotation = _args[1]
                elif _args[1] is _NoneType:
                    annotation = _args[0]
                else:  # pragma: no cover
                    # E.g. Union[A, B]
                    # we cover this  in tests coverage-tooling struggles here
                    continue
            else:  # pragma: no cover
                # E.g. Union[A, B, C]
                # we cover this in tests but coverage-tooling struggles here
                continue
        del _origin

        if not isinstance(annotation, type):
            # E.g. annotation = Tuple[int, int]
            # Tuple[int, int].__origin__ -> tuple
            caster = get_origin(annotation)
            if caster is None:  # pragma: no cover
                continue
        else:
            # E.g. annotation = tuple
            caster = annotation

        if (
            inspect.isclass(caster)
            and issubclass(caster, Sequence)
            and not issubclass(caster, str)  # strings don't need to be cast
            and caster is not list  # annotation is list to begin with
        ):
            if caster in {tuple, deque}:
                pass
            elif _is_namedtuple_type(annotation):

                def _unpack(x):
                    return annotation(*x)

                caster = _unpack
            else:  # pragma: no cover
                # covered in tests but coverage-tooling struggles here
                continue

            if param.kind is not _KEYWORD_ONLY:
                list_caster_by_pos[n] = caster
            if param.kind is not _POSITIONAL_ONLY:
                list_caster_by_name[name] = caster

    min_pos = min(list_caster_by_pos) if list_caster_by_pos else 0

    if not list_caster_by_name and not list_caster_by_pos:
        # no annotations associated with sequences
        return cast(_T, obj)

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if list_caster_by_pos and min_pos < len(args):
            # at least one positional argument needs to be cast
            args = list(args)
            for pos in list_caster_by_pos:
                if pos < len(args):
                    if isinstance(args[pos], (list, ListConfig)):
                        args[pos] = list_caster_by_pos[pos](args[pos])
                else:
                    # pos is inserted in dict in ascending order
                    break
        if list_caster_by_name and kwargs:
            named = set(list_caster_by_name).intersection(kwargs)
            for name in named:
                # at least one named-arg needs to be cast
                if isinstance(kwargs[name], (list, ListConfig)):
                    kwargs[name] = list_caster_by_name[name](kwargs[name])

        return obj(*args, **kwargs)

    return cast(_T, wrapper)
