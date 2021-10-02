import inspect
from functools import wraps
from typing import Callable, TypeVar, cast

import pydantic as _pyd

_T = TypeVar("_T", bound=Callable)

__all__ = ["validates_with_pydantic"]


def _validates_then_passes(obj):
    # `pydantic.validate_arguments` doesn't support instance-methods.
    # https://github.com/samuelcolvin/pydantic/issues/1222
    # and it doesn't seem like this is going to get fixed any time soon.
    #
    # That's okay! We can handle this ourselves!
    #
    # The nice thing about `validate_arguments` is that it exposes
    # a `.validate` method on the decorated object. We opt to use this
    # to perform the validation, and then simply call/instantiation `obj`
    # with the arguments as-is.
    #
    # This helps to ensure that the returned `obj(*arg, **kwargs)` will
    # never be affected by pydantic.
    #
    # NOTE: This means that we *do not* gain the use of pydantic's coercion
    # at this point.
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        target = obj.__init__
        NEEDS_SELF = True
    else:
        target = obj
        NEEDS_SELF = False

    pydantified = _pyd.validate_arguments(
        target, config={"arbitrary_types_allowed": True}
    )

    @wraps(obj)
    def wrapper(*args, **kwargs):
        if NEEDS_SELF:
            pydantified.validate(obj, *args, **kwargs)
        else:
            pydantified.validate(*args, **kwargs)
        return obj(*args, **kwargs)

    return wrapper


def validates_with_pydantic(func: _T) -> _T:  # pragma: no cover
    # NOTE: currently *does not* utilize pydantic's coercion mechanism
    # https://pydantic-docs.helpmanual.io/usage/models/#data-conversion
    #
    # It should be straightforward to expose a version of this that does,
    # but that will be riskier to use
    return cast(_T, _validates_then_passes(func))
