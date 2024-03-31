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


def prototype():

    from functools import wraps
    from typing import Any, Callable, Dict, Literal, Tuple

    import pydantic as pyd

    from hydra_zen import builds, instantiate as old_instantiate

    # note: this is super weird and busted when the callable is
    # `int` or `dict`
    # TODO: fix for builtins
    # TODO: handle __init__/__new__ gracefully: parse(Foo.__init__)(Foo, 'aa')
    #       -- maybe clone signature??
    parse = pyd.validate_call(config={"arbitrary_types_allowed": True})

    def _call_target(
        _target_: Callable[..., Any],
        _partial_: bool,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        full_key: str,
    ) -> Any:
        """Call target (type) with args and kwargs."""
        import functools

        from hydra._internal.instantiate._instantiate2 import (
            _convert_target_to_string,
            _extract_pos_args,
        )
        from hydra.errors import InstantiationException
        from omegaconf import OmegaConf

        try:
            args, kwargs = _extract_pos_args(args, kwargs)
            # detaching configs from parent.
            # At this time, everything is resolved and the parent link can cause
            # issues when serializing objects in some scenarios.
            for arg in args:
                if OmegaConf.is_config(arg):
                    arg._set_parent(None)
            for v in kwargs.values():
                if OmegaConf.is_config(v):
                    v._set_parent(None)
        except Exception as e:
            msg = (
                f"Error in collecting args and kwargs for '{_convert_target_to_string(_target_)}':"
                + f"\n{repr(e)}"
            )
            if full_key:
                msg += f"\nfull_key: {full_key}"

            raise InstantiationException(msg) from e

        if _partial_:
            try:
                return functools.partial(parse(_target_), *args, **kwargs)
            except Exception as e:
                msg = (
                    f"Error in creating partial({_convert_target_to_string(_target_)}, ...) object:"
                    + f"\n{repr(e)}"
                )
                if full_key:
                    msg += f"\nfull_key: {full_key}"
                raise InstantiationException(msg) from e
        else:
            try:
                print(args, kwargs)
                return parse(_target_)(*args, **kwargs)
            except Exception as e:
                msg = f"Error in call to target '{_convert_target_to_string(_target_)}':\n{repr(e)}"
                if full_key:
                    msg += f"\nfull_key: {full_key}"
                raise InstantiationException(msg) from e

    def loudly(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"Calling {fn.__name__}")
            return fn(*args, **kwargs)

        return wrapper

    def instantiate(x):
        try:
            from hydra._internal.instantiate import _instantiate2 as inst

            old = inst._call_target
            inst._call_target = loudly(_call_target)
            return old_instantiate(x)
        finally:
            inst._call_target = old

    def ff(x: tuple[int, int]):
        return x

    from dataclasses import dataclass

    @dataclass
    class A:
        x: "Literal[1] | A"

    instantiate(builds(A, x=builds(A, x=2)))


def parsing_stuff():
    import inspect

    def constructor_as_fn(cls):
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

        if not getattr(cls, "__annotations__", None):
            sig = inspect.signature(cls)
            wrapper_function.__annotations__ = {
                k: v.annotation for k, v in sig.parameters.items()
            }

        return wrapper_function

    import pydantic as pyd

    val = pyd.validate_call(
        config={"arbitrary_types_allowed": True}, validate_return=False
    )

    def get_signature(x: Any):
        try:
            return inspect.signature(x)
        except Exception:
            return None

    def with_pydantic_parsing(target):
        if inspect.isbuiltin(target):
            return target

        if inspect.isclass(target):
            if not (get_signature(target)):
                return target
            return val(constructor_as_fn(target))

        return val(target)

    class B:
        # __annotations__ = {"x": int}

        def __init__(self, x: int) -> None:
            self.x = x

        def __repr__(self) -> str:
            return f"B(x={self.x})"

        @classmethod
        def from_dict(cls, x: int):
            return cls(x)

        @staticmethod
        def from_dict2(x: int):
            return B(x)

    import dataclasses

    @dataclasses.dataclass
    class A:
        x: int | str

    def func(x):
        return x

    def bar(x: int):
        return x

    with_pydantic_parsing(func)(1)
    with_pydantic_parsing(bar)(1)
    with_pydantic_parsing(B.from_dict)(x=1)
    with_pydantic_parsing(B.from_dict2)(x=11)
    with_pydantic_parsing(A)(x=1)
    with_pydantic_parsing(B)(x=1)
