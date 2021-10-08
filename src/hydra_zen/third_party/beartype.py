# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from typing import Callable, TypeVar, cast

import beartype as bt

from hydra_zen._utils.coerce import coerce_sequences

_T = TypeVar("_T", bound=Callable)

__all__ = ["validates_with_beartype"]


def validates_with_beartype(obj: _T) -> _T:
    """Decorates a function or the init-method of an object with beartype [1]_.

    This can be used with

    Parameters
    ----------
    obj : Callable

    Returns
    -------
    obj_w_validation : Callable
        A wrapped function, or a class whose init-method has been
        wrapped in-place

    Notes
    -----
    ``beartype`` must be installed [2]_ as a separate dependency to leverage this validator.

    hydra-zen adds a sequence-coercion step that is not performed by beartype.
    This is implemented by `hydra_zen.experimental.coerce.coerce_sequences`. This
    coercion is necessary as Hydra can only read sequence data in as a list. I.e.
    without this coercion step, all non-list annotated sequence fields populated by
    Hydra would get roared at (raised) by beartype.

    It is recommended that `validates_with_beartype` be used in conjunction with
    the following `builds` settings:

      - ``hydra_convert="all"``: to ensure omegaconf containers are converted to std-lib types

      Please refer to beartype's documented list of compliances [3]_ to see what varieties of
      types it does and does not support.

    References
    ----------
    .. [1] https://github.com/beartype/beartype
    .. [2] https://github.com/beartype/beartype#install
    .. [3] https://github.com/beartype/beartype#compliance

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
    >>> validates_with_beartype(g)([1, 2, 3])  # input: list, output: tuple
    (1, 2, 3)
    """
    if inspect.isclass(obj) and hasattr(type, "__init__"):
        obj.__init__ = bt.beartype(obj.__init__)
        target = obj
    else:
        target = bt.beartype(obj)
    target = coerce_sequences(target)
    return cast(_T, target)
