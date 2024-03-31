# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
import inspect
from typing import Any, Callable, TypeVar, cast

import pydantic as _pyd

_T = TypeVar("_T", bound=Callable[..., Any])

__all__ = ["validates_with_pydantic"]


_default_validator = _pyd.validate_arguments(config={"arbitrary_types_allowed": True})


def validates_with_pydantic(
    obj: _T, *, validator: Callable[[_T], _T] = _default_validator
) -> _T:
    """Enables runtime type-checking of values, via the library ``pydantic``.

    I.e. ``obj = validates_with_beartype(obj)`` adds runtime type-checking
    to all calls of ``obj(*args, **kwargs)``, based on the type-annotations specified
    in the signature of ``obj``.

    This leverages ``pydantic.validate_arguments``, which is currently a
    beta [1]_ feature in pydantic.

    Parameters
    ----------
    obj : Callable

    validator : Type[pydantic.validate_arguments], optional
        A configured instance of pydantic's validation decorator.

        The default validator that we provide specifies:
           - arbitrary_types_allowed: True

    Returns
    -------
    obj_w_validation : Callable
        A wrapped function or a class whose init-method has been
        wrapped in-place

    See Also
    --------
    hydra_zen.third_party.beartype.validates_with_beartype

    Notes
    -----
    pydantic must be installed [2]_ as a separate dependency to leverage this validator.
    Using ``validates_with_pydantic`` as a ``zen_wrapper`` will create a dependency on
    pydantic among resulting yamls: these yamls will also be validated by pydantic
    upon instantiation.

    It is recommended that `validates_with_pydantic` be used in conjunction with
    the following `builds` settings:

      - ``hydra_convert="all"``: to ensure omegaconf containers are converted to std-lib types

    Users should be aware of pydantic's data conversion strategy [3]_; pydantic
    may cast data so that it will conform to its annotated type.

    References
    ----------
    .. [1] https://pydantic-docs.helpmanual.io/usage/validation_decorator/
    .. [2] https://pydantic-docs.helpmanual.io/install/
    .. [3] https://pydantic-docs.helpmanual.io/usage/models/#data-conversion

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen.third_party.pydantic import validates_with_pydantic
    >>> from pydantic import PositiveInt

    >>> def f(x: PositiveInt): return x

    >>> f(-100)  # bad value passes
    -100

    >>> val_f = validates_with_pydantic(f)  # f + validation
    >>> val_f(-100)  # bad value gets caught
    ValidationError: 1 validation error for F (...)

    Applying `validates_with_pydantic` to a class-object will wrap its ``__init__``
    method in-place.

    >>> class A:
    ...     def __init__(self, x: PositiveInt): ...
    >>> validates_with_pydantic(A)  # wrapping occurs in-place
    __main__.A
    >>> A(-10)
    ValidationError: 1 validation error for Init

    **Adding pydantic validation to configs**

    This is designed to be used with the ``zen_wrappers`` feature of `builds`.

    >>> from hydra_zen import builds, instantiate
    >>> # instantiations of `conf` will be validated by pydantic
    >>> conf = builds(
    ...     f,
    ...     zen_wrappers=validates_with_pydantic,
    ...     # recommended builds-settings for pydantic-validation
    ...     populate_full_signature=True,
    ...     hydra_convert="all",
    ... )
    >>> instantiate(conf, x=10)
    10
    >>> instantiate(conf, x=-2)
    ValidationError: 1 validation error for F (...)

    Note that pydantic's data-coercion ensures smooth compatibility with Hydra.
    I.e. lists will be coerced to the appropriate annotated sequence type.

    >>> def g(x: tuple): return x  # note the annotation
    >>> validates_with_pydantic(g)([1, 2, 3])  # input: list, output: tuple
    (1, 2, 3)

    Consider using :func:`~hydra_zen.make_custom_builds_fn` to add validation to
    all configs.
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


def _constructor_as_fn(cls):
    """Makes a shim around a class constructor so that it is compatible with pydantic validation.

    Notes
    -----
    `pydantic.validate_call` mishandles class constructors; it expects that
    `cls`/`self` should be passed explicitly to the constructor. This shim
    corrects that.
    """

    @functools.wraps(cls)
    def wrapper_function(*args, **kwargs):
        return cls(*args, **kwargs)

    annotations = getattr(cls, "__annotations__", {})

    # In a case like:
    # class A:
    #   x: int
    #   def __init__(self, y: int): ...
    #
    #  y will not be in __annotations__ but it should be in the signature,
    #  so we add it to the annotations.

    sig = inspect.signature(cls)
    for p, v in sig.parameters.items():
        if p not in annotations:
            annotations[p] = v.annotation
    wrapper_function.__annotations__ = annotations

    return wrapper_function


if _pyd.__version__ >= "2":
    _validator = _pyd.validate_call(
        config={"arbitrary_types_allowed": True}, validate_return=False
    )
else:
    _validator = _pyd.validate_arguments(
        config={"arbitrary_types_allowed": True, "validate_return": False}
    )


def _get_signature(x: Any):
    try:
        return inspect.signature(x)
    except Exception:
        return None


def with_pydantic_parsing(target: _T) -> _T:
    if inspect.isbuiltin(target):
        return target

    if not (_get_signature(target)):
        return target

    if inspect.isclass(target):
        return cast(_T, _validator(_constructor_as_fn(target)))

    return _validator(target)
