# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing import Any, Callable, FrozenSet, Optional, Type, TypeVar, Union, overload

from typing_extensions import Literal

from hydra_zen.typing import Builds, DataclassOptions, Just
from hydra_zen.typing._implementations import _HydraPrimitive  # type: ignore
from hydra_zen.typing._implementations import _SupportedViaBuilds  # type: ignore
from hydra_zen.typing._implementations import DataClass_, ZenConvert

from ._implementations import ConfigComplex, DefaultBuilds

# pyright: strict
T = TypeVar("T")
TD = TypeVar("TD", bound=DataClass_)
TC = TypeVar("TC", bound=Callable[..., Any])
TP = TypeVar("TP", bound=_HydraPrimitive)
TB = TypeVar("TB", bound=Union[_SupportedViaBuilds, FrozenSet[Any]])

__all__ = ["just"]


@overload
def just(
    obj: TP,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> TP:
    ...


@overload
def just(
    obj: complex,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> ConfigComplex:
    ...


@overload
def just(
    obj: TC,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> Just[TC]:
    ...


@overload
def just(
    obj: TB,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> Builds[Type[TB]]:
    ...


@overload
def just(
    obj: TD,
    *,
    zen_convert: Literal[None] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> Type[Builds[Type[TD]]]:
    ...


@overload
def just(
    obj: DataClass_,
    *,
    zen_convert: ZenConvert,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> Any:
    ...


@overload
def just(
    obj: Any,
    *,
    zen_convert: Optional[ZenConvert] = ...,
    hydra_recursive: Optional[bool] = ...,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = ...,
    zen_dataclass: Optional[DataclassOptions] = ...,
) -> Any:
    ...


def just(
    obj: Any,
    *,
    zen_convert: Optional[ZenConvert] = None,
    hydra_recursive: Optional[bool] = None,
    hydra_convert: Optional[Literal["none", "partial", "all", "object"]] = None,
    zen_dataclass: Optional[DataclassOptions] = None,
) -> Any:
    """`just(obj)` returns a config that, when instantiated, just returns `obj`.

    I.e., `instantiate(just(obj)) == obj`

    `just` is designed to be idempotent. I.e., `just(obj) == just(just(obj))`

    Parameters
    ----------
    obj : Callable[..., Any] | HydraSupportedPrimitive | ZenSupportedPrimitive
        A type (e.g. a class-object), function-object, or a value that is either
        supported by Hydra or has auto-config support via hydra-zen.

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

    hydra_convert : Optional[Literal["none", "partial", "all", "object"]], optional (default="none")
        Determines how Hydra treats the non-primitive, omegaconf-specific objects
        during instantiateion [3]_.

        - ``"none"``: No conversion occurs; omegaconf containers are passed through (Default)
        - ``"partial"``: ``DictConfig`` and ``ListConfig`` objects converted to ``dict`` and
          ``list``, respectively. Structured configs and their fields are passed without conversion.
        - ``"all"``: All passed objects are converted to dicts, lists, and primitives, without
          a trace of OmegaConf containers.
        - ``"object"``: Passed objects are converted to dict and list. Structured Configs are converted to instances of the backing dataclass / attr class.

        If ``None``, the ``_convert_`` attribute is not set on the resulting config.

    zen_dataclass : Optional[DataclassOptions]
        A dictionary that can specify any option that is supported by
        :py:func:`dataclasses.make_dataclass` other than `fields`.
        The default value for `unsafe_hash` is `True`.

        These options are only relevant when the input to `just` is a dataclass
        instance. Otherwise, `just` does not utilize these options when auto-generating
        configs.

        Additionally, the `module` field can be specified to enable pickle
        compatibility. See `hydra_zen.typing.DataclassOptions` for details.

    Returns
    -------
    out : HydraSupportedPrimitive | Builds[Type[obj]]
        ``out`` is ``obj`` unchanged if ``obj`` is supported natively by Hydra, otherwise ``out`` is a dynamically-generated dataclass type or instance.

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
    Here, a "config" is a dynamically-generated dataclass type that is designed to be
    compatible with Hydra [1]_.

    The configs produced by ``just(<type_or_func>)`` introduce an explicit dependency
    on hydra-zen. I.e. hydra-zen must be installed in order to instantiate any config
    that used `just`.

    Examples
    --------
    **Basic usage**

    >>> from hydra_zen import just, instantiate, to_yaml

    `just`, called on a value of a type that is natively supported by Hydra,
    will return that value unchanged:

    >>> just(1)
    1
    >>> just({"a": False})
    {"a": False}

    `just` can be used to create a config that will simply import a class-object or function when the config is instantiated by Hydra.


    >>> class A: pass
    >>> def my_func(x: int): pass

    >>> instantiate(just(A)) is A
    True
    >>> instantiate(just(my_func)) is my_func
    True

    `just` dynamically generates dataclass types, a.k.a structured configs, to describe
    ``obj``

    >>> just(A)
    Just_A(_target_='hydra_zen.funcs.get_obj', path='__main__.A')
    >>> just(my_func)
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
    >>> instantiate(just({1, 2, 3}))
    {1, 2, 3}

    By default, `just` will convert a dataclass instance to a corresponding targeted
    config. This behavior can be modified via :ref:`zen-convert settings<zen-convert>`.

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class B:
    ...     x: complex
    >>>
    >>> instantiate(just(B)) is B
    True
    >>> instantiate(just(B(2+3j))) == B(2+3j)
    True

    `just` operates recursively within sequences, mappings, and dataclass fields.

    >>> z_dict = {'a': [3-4j, 1+2j]}
    >>> assert instantiate(just(z_dict)) == z_dict
    >>>
    >>> from typing import Any
    >>> @dataclass
    ... class C:
    ...     x: Any
    >>>
    >>> instantiate(just(C(B(2+3j))))
    C(B(2+3j))

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
    return DefaultBuilds.just(**locals())
