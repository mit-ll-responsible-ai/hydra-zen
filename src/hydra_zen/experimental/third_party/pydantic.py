import inspect
from typing import Callable, TypeVar, cast

import pydantic as _pyd

_T = TypeVar("_T", bound=Callable)

__all__ = ["validates_with_pydantic"]


_default_validator = _pyd.validate_arguments(config={"arbitrary_types_allowed": True})


def validates_with_pydantic(
    obj: _T, validator: Callable[[_T], _T] = _default_validator
) -> _T:
    """Decorates a function or the init-method of an objects with a pydantic
    validation decorator.

    This leverages `pydantic.validate_arguments`, which is currently in Beta [1]_.

    Parameters
    ----------
    obj : Callable

    validator : pydantic.validate_arguments, optional
        A configured instance of pydantic's validation decorator.
        The default validator that we provide specifies:
           - arbitrary_types_allowed: True

    Returns
    -------
    obj_w_validation : Callable
        A wrapped function or a class whose init-method has been
        wrapped in-place

    Notes
    -----
    Users should be aware of pydantic's data conversion strategy [2]_; pydantic
    may cast data so that it will conform to its annotated type.

    References
    ----------
    .. [1] https://pydantic-docs.helpmanual.io/usage/validation_decorator/
    .. [2] https://pydantic-docs.helpmanual.io/usage/models/#data-conversion

    Examples
    --------
    >>> from pydantic import PositiveInt
    >>> def f(x: PositiveInt): return x
    >>> val_f = validates_with_pydantic(needs_pos_int)
    >>> f(-100)
    -100
    >>> val_f(-100)
    ValidationError: 1 validation error for F [...]

    >>> class A:
    ...     def __init__(self, x: PositiveInt): ...
    >>> validates_with_pydantic(A)  # wrapping occurs in-place
    __main__.A
    >>> A(-10)
    ValidationError: 1 validation error for Init
    """
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        if hasattr(obj.__init__, "validate"):
            # already decorated by pydantic
            return cast(_T, obj)
        obj.__init__ = validator(obj.__init__)
    else:
        if hasattr(obj, "validate"):
            # already decorated by pydantic
            return cast(_T, obj)
        obj = cast(_T, validator(obj))

    return cast(_T, obj)
