# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import Any, FrozenSet, Type, TypeVar, Union, overload

from hydra_zen.typing import Builds, Importable, Just
from hydra_zen.typing._implementations import _HydraPrimitive  # type: ignore
from hydra_zen.typing._implementations import _SupportedViaBuilds  # type: ignore

from ._implementations import sanitized_default_value
from ._value_conversion import ConfigComplex

# pyright: strict

TP = TypeVar("TP", bound=_HydraPrimitive)
TB = TypeVar("TB", bound=Union[_SupportedViaBuilds, FrozenSet[Any]])

__all__ = ["just"]


@overload
def just(obj: TP) -> TP:  # pragma: no cover
    ...


@overload
def just(obj: complex) -> ConfigComplex:  # pragma: no cover
    ...


@overload
def just(obj: TB) -> Builds[Type[TB]]:  # pragma: no cover
    ...


@overload
def just(obj: Importable) -> Type[Just[Importable]]:  # pragma: no cover
    ...


@overload
def just(obj: Any) -> Any:  # pragma: no cover
    ...


def just(obj: Any) -> Any:
    """Returns a structured config that "just" describes the specified value.

    Parameters
    ----------
    obj : Type | Callable[..., Any] HydraSupportedPrimitive | ZenSupportedPrimitive
        A type (e.g. a class-object), function-object, or a value of a type that is
        supported by either  hydra-zen or Hydra.

    Returns
    -------
    config : HydraSupportedPrimitive | Builds
        A structured config that describes ``obj``.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    make_config: Creates a general config with customized field names, default values, and annotations.

    Notes
    -----
    The configs produced by ``just(<type_or_func>)`` introduce an explicit dependency
    on hydra-zen. I.e. hydra-zen must be installed in order to instantiate any config
    that used `just`.

    hydra-zen provides specialized support for values of the following types:

    - :py:class:`bytes`
    - :py:class:`bytearray`
    - :py:class:`complex`
    - :py:class:`collections.Counter`
    - :py:class:`collections.deque`
    - :py:func:`functools.partial`
    - :py:class:`pathlib.Path`
    - :py:class:`pathlib.PosixPath`
    - :py:class:`pathlib.WindowsPath`
    - :py:class:`range`
    - :py:class:`set`
    - :py:class:`frozenset`

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import just, instantiate, to_yaml

    Using ``just`` to create a config that returns an object – without calling it –
    upon instantiation.

    >>> class A: pass
    >>> Conf_A = just(A)  # returns a dataclass-type
    >>> instantiate(Conf_A) is A
    True

    >>> def my_func(x: int): pass
    >>> Conf_func = just(my_func)  # returns a dataclass-type
    >>> instantiate(Conf_func) is my_func
    True


    Calling ``just`` on a value of a type that is natively supported by Hydra
    will reurn that value unchanged

    >>> just(1)
    1
    >>> just(False)
    False

    Calling ``just`` on a value of a type that has special support from hydra-zen
    will return a structured config instance that, when instantiated, returns that
    value

    >>> just(1+2j)
    ConfigComplex(real=1.0, imag=2.0, _target_='builtins.complex')

    >>> instantiate(just(1+2j))
    (1+2j)

    >>> just({1, 2, 3})
    Builds_set(_target_='builtins.set', _args_=((1, 2, 3),))

    >>> instantiate(just({1, 2, 3})
    {1, 2, 3}

    ``just`` operates recursively within sequences and mappings.

    >>> just({'a': [3-4j, 1+2j]})
    'a': [ConfigComplex(real=3.0, imag=-4.0, _target_='builtins.complex'),
         ConfigComplex(real=1.0, imag=2.0, _target_='builtins.complex')]}

    **Auto-Application of just**

    Both ``builds`` and ``make_config`` will automatically (and recursively) apply
    ``just`` to all configured values. E.g. in the following example `just` will be
    applied to both  the complex-valued the list and to ``sum``.

    >>> from hydra_zen import make_config
    >>> Conf2 = make_config(data=[1+2j, 2+3j], reduction_fn=sum)

    >>> print(to_yaml(Conf2))
    data:
    - real: 1.0
      imag: 2.0
      _target_: builtins.complex
    - real: 2.0
      imag: 3.0
      _target_: builtins.complex
    reduction_fn:
      _target_: hydra_zen.funcs.get_obj
      path: builtins.sum

    >>> conf = instantiate(Conf2)
    >>> conf.reduction_fn(conf.data)
    (3+5j)
    """
    return sanitized_default_value(
        obj,
        allow_zen_conversion=True,
        structured_conf_permitted=True,
        field_name="",
        error_prefix="",
    )
