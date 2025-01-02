# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: reportUnnecessaryTypeIgnoreComment=false
import functools
import inspect
from typing import Any, Callable, TypeVar, Union, cast

import pydantic as _pyd

_T = TypeVar("_T", bound=Callable[..., Any])

__all__ = ["validates_with_pydantic"]


if _pyd.__version__ >= "2.0":  # pragma: no cover
    _default_parser = _pyd.validate_call(
        config={"arbitrary_types_allowed": True}, validate_return=False  # type: ignore
    )
else:  # pragma: no cover
    _default_parser = _pyd.validate_arguments(
        config={"arbitrary_types_allowed": True, "validate_return": False}  # type: ignore
    )


def _constructor_as_fn(cls: Any) -> Any:
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


def _get_signature(x: Any) -> Union[None, inspect.Signature]:
    try:
        return inspect.signature(x)
    except Exception:
        return None


def pydantic_parser(target: _T, *, parser: Callable[[_T], _T] = _default_parser) -> _T:
    """A target-wrapper that adds pydantic parsing to the target.

    This can be passed to `instantiate` as a `_target_wrapper_` to add pydantic parsing
    to the (recursive) instantiation of the target.

    Parameters
    ----------
    target : Callable

    parser : Type[pydantic.validate_arguments], optional
        A configured instance of pydantic's validation decorator.

        The default validator that we provide specifies:
           - arbitrary_types_allowed: True

    Examples
    --------
    .. code-block:: python

       from hydra_zen import builds, instantiate
       from hydra_zen.third_party.pydantic import pydantic_parser

       from pydantic import PositiveInt

       def f(x: PositiveInt): return x

       good_conf = builds(f, x=10)
       bad_conf = builds(f, x=-3)

    >>> instantiate(good_conf, _target_wrapper_=pydantic_parser)
    10
    >>> instantiate(bad_conf, _target_wrapper_=pydantic_parser)
    ValidationError: 1 validation error for f (...)

    This also enables type conversion / parsing. E.g. Hydra can
    only produce lists from the CLI, but this parsing layer can
    convert them based on the annotated type. (Note: this only
    works for pydantic v2 and higher.)

    >>> def g(x: tuple): return x
    >>> conf = builds(g, x=[1, 2, 3])
    >>> instantiate(conf, _target_wrapper_=pydantic_parser)
    (1, 2, 3)
    """
    if inspect.isbuiltin(target):
        return cast(_T, target)

    if isinstance(target, type) and issubclass(target, _pyd.BaseModel):
        # this already applies pydantic parsing
        return cast(_T, target)

    if not (_get_signature(target)):
        return cast(_T, target)

    if inspect.isclass(target):
        return cast(_T, parser(_constructor_as_fn(target)))

    return parser(target)


def validates_with_pydantic(
    obj: _T, *, validator: Callable[[_T], _T] = _default_parser
) -> _T:
    """
    .. deprecated:: 0.13.0

       Use `hydra_zen.third_party.pydantic.pydantic_parser` instead.
    """
    return pydantic_parser(obj, parser=validator)
