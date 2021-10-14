# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from typing import Callable, TypeVar, cast

import beartype as bt

from hydra_zen._utils.coerce import coerce_sequences

_T = TypeVar("_T", bound=Callable)

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

    Notes
    -----
    ``beartype`` [1]_ must be installed as a separate dependency to leverage this validator.
    Using ``validates_with_beartype`` as a ``zen_wrapper`` will create a dependency on
    beartype among resulting yamls as well, these yamls will also be validated by beartype
    upon instantiation.

    Please refer to beartype's documentation [2]_ to see what varieties of types it does and
    does not support.

    It is recommended that `validates_with_beartype` be used in conjunction with
    the following `builds` settings:

    - ``hydra_convert="all"``: to ensure omegaconf containers are converted to std-lib types

    **Data Coercion Behavior**

    hydra-zen adds a data-coercion step that is not performed by ``beartype``.
    This only impacts fields in ``obj`` annotated with a (non-string) sequence-type
    annotation, which are passed list-type data. All other fields rely solely on beartype's
    native behavior.

    E.g. a field with a ``Tuple``-annotation, if passed a list, will see that list be cast
    to a tuple. See the Examples section for more details.

    This coercion is necessary as Hydra can only read (non-string) sequential data
    from a config as a list. I.e. without this coercion step, all non-list/string annotated
    sequence fields populated by Hydra would get rasied (roared at) by beartype. Ultimately,
    this data-coercion strategy is designed to be as minimalistic as possible while ensuring
    that type-checked interfaces will be compatible with Hydra.

    References
    ----------
    .. [1] https://github.com/beartype/beartype
    .. [2] https://github.com/beartype/beartype#compliance

    Examples
    --------
    >>> from hydra_zen.third_party.beartype import validates_with_beartype
    >>> from beartype.cave import ScalarTypes
    >>> def f(x: ScalarTypes): return x  # a scalar is any real-valued number
    >>> val_f = validates_with_beartype(f)
    >>> f([1, 2])  # doesn't catch bad input
    [1, 2]
    >>> val_f([1, 2])
    BeartypeCallHintPepParamException: @beartyped f() parameter x=[1, 2] violates type hint [...]

    >>> class A:
    ...     def __init__(self, x: ScalarTypes): ...
    >>> validates_with_beartype(A)  # wrapping occurs in-place
    __main__.A
    >>> A([1, 2])
    BeartypeCallHintPepParamException: @beartyped A.__init__() parameter x=[1, 2] violates type hint [...]

    This is designed to be used with the ``zen_wrappers`` feature of `builds`.

    >>> from hydra_zen import builds, instantiate
    >>> # instantiations of `conf` will be validated by beartype
    >>> conf = builds(f, populate_full_signature=True, zen_wrappers=validates_with_beartype)
    >>> instantiate(conf, x=10)  # 10 is a scalar: ok!
    10
    >>> instantiate(conf, x=[1, 2])  # [1, 2] is not a scalar: roar!
    BeartypeCallHintPepParamException: @beartyped f() parameter x=[1, 2] violates type hint [...]

    Note that sequence-coercion is enabled to ensure smooth compatibility with Hydra.

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
