# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from typing import Any, Callable, TypeVar, cast

import beartype as bt

from hydra_zen._utils.coerce import coerce_sequences

_T = TypeVar("_T", bound=Callable[..., Any])

__all__ = ["validates_with_beartype"]


def validates_with_beartype(obj: _T) -> _T:
    """Enables runtime type-checking of values, via the library ``beartype``.

    I.e. ``obj = validates_with_beartype(obj)`` adds runtime type-checking
    to all calls of ``obj(*args, **kwargs)``, based on the type-annotations
    specified in the signature of ``obj``.

    This is designed to be used as a "zen-wrapper"; see Examples for details.

    Parameters
    ----------
    obj : Callable

    Returns
    -------
    obj_w_validation : Callable
        A wrapped function, or a class whose init-method has been
        wrapped in-place.

    See Also
    --------
    hydra_zen.third_party.pydantic.validates_with_pydantic

    Notes
    -----
    ``beartype`` [1]_ must be installed as a separate dependency to leverage this
    validator. Refer to beartype's documentation [2]_ to see what varieties of types it
    does and does not support.

    Using ``validates_with_beartype`` as a ``zen_wrapper`` will create a dependency on
    beartype among resulting yamls: these yamls will also be validated by
    beartype upon instantiation.

    **Data-Coercion Behavior**

    hydra-zen adds a data-coercion step that is not performed by ``beartype``.
    This only impacts fields in ``obj`` annotated with a (non-string) sequence-type
    annotation, which are passed list-type data. All other fields rely solely on
    beartype's native behavior.

    E.g. a field with a ``Tuple``-annotation, if passed a list, will see that list be
    cast to a tuple. See the Examples section for more details.

    References
    ----------
    .. [1] https://github.com/beartype/beartype
    .. [2] https://github.com/beartype/beartype#compliance

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen.third_party.beartype import validates_with_beartype

    >>> from beartype.cave import ScalarTypes
    >>> def f(x: ScalarTypes): return x  # a scalar is any real-valued number
    >>> f([1, 2])  # doesn't catch bad input
    [1, 2]

    >>> val_f = validates_with_beartype(f)  # f + validation
    >>> val_f([1, 2])
    BeartypeCallHintPepParamException: @beartyped f() parameter x=[1, 2] violates type hint [...]

    Applying `validates_with_beartype` to a class-object will wrap its ``__init__``
    method in-place.

    >>> class A:
    ...     def __init__(self, x: ScalarTypes): ...
    >>> validates_with_beartype(A)  # wrapping occurs in-place
    __main__.A
    >>> A([1, 2])
    BeartypeCallHintPepParamException: @beartyped A.__init__() parameter x=[1, 2] violates type hint [...]

    **Adding beartype validation to configs**

    This is designed to be used with the ``zen_wrappers`` feature of `builds`.

    >>> from hydra_zen import builds, instantiate

    >>> Conf = builds(f, populate_full_signature=True, zen_wrappers=validates_with_beartype)

    Instantiating ``Conf`` will prompt ``beartype`` to check the types of configured
    parameters against the corresponding annotations on ``f``.

    >>> instantiate(Conf, x=10)  # 10 is a scalar: ok!
    10
    >>> instantiate(Conf, x=[1, 2])  # [1, 2] is not a scalar: roar!
    BeartypeCallHintPepParamException: @beartyped f() parameter x=[1, 2] violates type hint [...]

    Consider using :func:`~hydra_zen.make_custom_builds_fn` to add validation to
    all configs.

    **Sequence-coercion behavior for compatibility with Hydra**

    Note that sequence-coercion is enabled to ensure smooth compatibility with Hydra,
    as Hydra will interpret all (non-string) sequential data structures as lists.

    >>> def g(x: tuple): return x  # note the annotation
    >>> g([1, 2, 3])
    [1, 2, 3]
    >>> validates_with_beartype(g)([1, 2, 3])  # input: list, output: tuple
    (1, 2, 3)

    Only inputs of type list and ``ListConfig`` get cast in this way, since Hydra will
    read non-string sequential data from configs as either of these two types

    >>> validates_with_beartype(g)({1, 2, 3})  # input: a set
    BeartypeCallHintPepParamException: @beartyped g() parameter x={1, 2, 3} violates type hint [...]
    """
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        obj.__init__ = bt.beartype(obj.__init__)
        target = obj
    else:
        target = bt.beartype(obj)
    target = coerce_sequences(target)
    return cast(_T, target)
