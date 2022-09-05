# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from typing import Any, FrozenSet, Optional, Type, TypeVar, Union, overload

from typing_extensions import Literal

from hydra_zen.structured_configs import _utils
from hydra_zen.typing import Builds, Importable, Just
from hydra_zen.typing._implementations import _HydraPrimitive  # type: ignore
from hydra_zen.typing._implementations import _SupportedViaBuilds  # type: ignore
from hydra_zen.typing._implementations import (
    AllConvert,
    DataClass_,
    InstOrType,
    ZenConvert,
)

from ._implementations import sanitized_default_value
from ._utils import merge_settings
from ._value_conversion import ConfigComplex

# pyright: strict
TD = TypeVar("TD", bound=DataClass_)
TP = TypeVar("TP", bound=_HydraPrimitive)
TB = TypeVar("TB", bound=Union[_SupportedViaBuilds, FrozenSet[Any]])

__all__ = ["just"]


_JUST_CONVERT_SETTINGS = AllConvert(dataclass=True)


@overload
def just(
    obj: TP,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> TP:  # pragma: no cover
    ...


@overload
def just(
    obj: complex,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> ConfigComplex:  # pragma: no cover
    ...


@overload
def just(
    obj: TB,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> Builds[Type[TB]]:  # pragma: no cover
    ...


@overload
def just(
    obj: InstOrType[TD],
    *,
    zen_convert: Literal[None] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> Type[Builds[Type[TD]]]:  # pragma: no cover
    ...


@overload
def just(
    obj: Importable,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> Type[Just[Importable]]:  # pragma: no cover
    ...


@overload
def just(
    obj: Any,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = ...,
) -> Any:  # pragma: no cover
    ...


def just(
    obj: Any,
    *,
    zen_convert: Optional[ZenConvert] = None,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all"]] = None,
) -> Any:
    """`just(obj)` returns a config that, when instantiated, just returns `obj`.

    `instantiate(just(obj)) == obj`

    `just` is designed to be idempotent: `just(obj) == just(just(obj))`

    Parameters
    ----------
    obj : Type | Callable[..., Any] HydraSupportedPrimitive | ZenSupportedPrimitive
        A value, type (e.g. a class-object), or function-object that is supported by
        either  hydra-zen or Hydra.

    zen_convert : Optional[ZenConvert]
        A dictionary that modifies hydra-zen's value and type conversion behavior.
        Consists of the following optional key-value pairs (:ref:`zen-convert`):

        - `dataclass` : `bool` (default=True):
            If `True` any dataclass type/instance without a
            `_target_` field is automatically converted to a targeted config
            that will instantiate to that type/instance. Otherwise the dataclass
            type/instance will be passed through as-is.


    hydra_recursive : Optional[bool], optional (default=True)
        If ``True``, then Hydra will recursively instantiate all other
        hydra-config objects nested within this config [2]_.

        If ``None``, the ``_recursive_`` attribute is not set on the resulting config.

    hydra_convert : Optional[Literal["none", "partial", "all"]], optional (default="none")
        Determines how Hydra handles the non-primitive, omegaconf-specific objects passed to
        ``<hydra_target>`` [3]_.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
          a trace of OmegaConf containers.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.


    Returns
    -------
    config : HydraSupportedPrimitive | Builds
        A structured config that describes ``obj``.

    Raises
    ------
    hydra_zen.errors.HydraZenUnsupportedPrimitiveError

    References
    ----------
    .. [1] https://hydra.cc/docs/tutorials/structured_config/intro/
    .. [2] https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation
    .. [3] https://hydra.cc/docs/advanced/instantiate_objects/overview/#parameter-conversion-strategies

    See Also
    --------
    builds : Create a targeted structured config designed to "build" a particular object.
    make_config: Creates a general config with customized field names, default values, and annotations.

    Notes
    -----
    A "config" is a dynamically-generated dataclass type that is designed to be
    compatible with Hydra [1]_.

    The configs produced by ``just(<type_or_func>)`` introduce an explicit dependency
    on hydra-zen. I.e. hydra-zen must be installed in order to instantiate any config
    that used `just`.

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import just, instantiate, to_yaml

    Calling `just` on a value of a type that is natively supported by Hydra
    will reurn that value unchanged

    >>> just(1)
    1
    >>> just(False)
    False

    `just` can be used to create a config that will simply import a class-object or function when the config is instantiated by Hydra.


    >>> class A: pass
    >>> def my_func(x: int): pass

    >>> instantiate(just(A)) is A
    True
    >>> instantiate(just(my_func)) is my_func
    True

    >>> just(A)()
    Just_A(_target_='hydra_zen.funcs.get_obj', path='__main__.A')
    >>> just(my_func)()
    Just_my_func(_target_='hydra_zen.funcs.get_obj', path='__main__.my_func')

    **Auto-config support**

    Calling `just` on a value of a type that has
    :ref:`special support from hydra-zen <additional-types>`
    will return a structured config instance that, when instantiated by Hydra, returns
    that value.

    >>> just(1+2j)
    ConfigComplex(real=1.0, imag=2.0, _target_='builtins.complex')
    >>> instantiate(just(1+2j))
    (1+2j)

    >>> just({1, 2, 3})
    Builds_set(_target_='builtins.set', _args_=((1, 2, 3),))
    >>> instantiate(just({1, 2, 3})
    {1, 2, 3}

    `just` operates recursively within sequences and mappings.

    By default, `just` will convert a dataclass instance (including pydantic
    dataclasses) to a targeted config that will recreate the instance upon
    Hydra-instantiation. This enables users to leverage annotations and datatypes
    that are not supported directly by Hydra.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class B:
    ...     x: complex
    >>>
    >>> Conf = just(B(x=2+3j))
    >>> Conf._target_, Conf.x
    (__main__.B, ConfigComplex(real=2.0, imag=3.0, _target_='builtins.complex'))
    >>> instantiate(Conf)
    B(x=(2+3j))

    >>> z_dict = {'a': [3-4j, 1+2j]}
    >>> just(z_dict)
    'a': [ConfigComplex(real=3.0, imag=-4.0, _target_='builtins.complex'),
         ConfigComplex(real=1.0, imag=2.0, _target_='builtins.complex')]}
    >>> assert instantiate(just(z_dict)) == z_dict

    This behavior can be modified via :ref:`zen-convert settings<zen-convert>`.

    **Auto-Application of just**

    Both `builds` and `make_config` will automatically (and recursively) apply
    `just` to all configured values. E.g. in the following example `just` will be
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
    convert_settings = merge_settings(zen_convert, _JUST_CONVERT_SETTINGS)
    del zen_convert
    _utils.validate_hydra_options(
        hydra_recursive=hydra_recursive, hydra_convert=hydra_convert
    )

    return sanitized_default_value(
        obj,
        allow_zen_conversion=True,
        structured_conf_permitted=True,
        field_name="",
        error_prefix="",
        convert_dataclass=convert_settings["dataclass"],
        hydra_convert=hydra_convert,
        hydra_recursive=hydra_recursive,
    )
