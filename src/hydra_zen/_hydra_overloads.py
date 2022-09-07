# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
"""
Provides annotation overloads for various hydra functions, using the types defined in `hydra_zen.typing`.
This enables tools like IDEs to be more incisive during static analysis and to provide users with additional
context about their code.

E.g.

.. code::

   from hydra_zen import builds, instantiate
   DictConfig = builds(dict, a=1, b=2)  # type: Type[Builds[Type[dict]]]

   # static analysis tools can provide useful type information
   # about the object that is instantiated from the config
   out = instantiate(DictConfig)  # type: dict

"""

# pyright: strict

import pathlib
from dataclasses import is_dataclass
from functools import wraps
from typing import IO, Any, Callable, Type, TypeVar, Union, cast, overload

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .structured_configs._just import just
from .structured_configs._value_conversion import ConfigComplex, ConfigPath
from .typing import Builds, Just, Partial
from .typing._implementations import DataClass_, HasTarget, InstOrType, IsPartial

__all__ = ["instantiate", "to_yaml", "save_as_yaml", "load_from_yaml", "MISSING"]


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class _TightBind:  # pragma: no cover
    ...


@overload
def instantiate(
    config: _TightBind, *args: Any, **kwargs: Any
) -> Any:  # pragma: no cover
    ...


@overload
def instantiate(
    config: InstOrType[ConfigPath], *args: Any, **kwargs: Any
) -> pathlib.Path:  # pragma: no cover
    ...


@overload
def instantiate(
    config: InstOrType[ConfigComplex], *args: Any, **kwargs: Any
) -> complex:  # pragma: no cover
    ...


@overload
def instantiate(
    config: InstOrType[Just[T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: InstOrType[IsPartial[Callable[..., T]]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: InstOrType[Builds[Callable[..., T]]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[HasTarget, ListConfig, DictConfig, DataClass_, Type[DataClass_]],
    *args: Any,
    **kwargs: Any
) -> Any:  # pragma: no cover
    ...


def instantiate(config: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Instantiates the target of a targeted config.

    This is an alias of :func:`hydra.utils.instantiate` [1]_.

    By default, `instantiate` will recursively instantiate nested configurations [1]_.

    Parameters
    ----------
    config : Builds[Type[T] | Callable[..., T]]
        The targeted config whose target will be instantiated/called.

    *args: Any
        Override values, specified by-position. Take priority over
        the positional values provided by ``config``.

    **kwargs : Any
        Override values, specified by-name. Take priority over
        the named values provided by ``config``.

    Returns
    -------
    instantiated : T
        The instantiated target. Instantiated using the values provided
        by ``config`` and/or overridden via ``*args`` and ``**kwargs``.

    See Also
    --------
    builds: Returns a config, which describes how to instantiate/call ``<hydra_target>``.
    just: Produces a config that, when instantiated by Hydra, "just" returns the un-instantiated target-object

    Notes
    -----
    This is an alias for ``hydra.utils.instantiate``, but adds additional static type
    information.

    During instantiation, Hydra performs runtime validation of data based on a limited set of
    type-annotations that can be associated with the fields of the provided config [2]_ [3]_.

    Hydra supports a string-based syntax for variable interpolation, which enables configured
    values to be set in a self-referential and dynamic manner [4]_.

    References
    ----------
    .. [1] https://hydra.cc/docs/advanced/instantiate_objects/overview
    .. [2] https://omegaconf.readthedocs.io/en/latest/structured_config.html#simple-types
    .. [3] https://omegaconf.readthedocs.io/en/latest/structured_config.html#runtime-type-validation-and-conversion
    .. [4] https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation

    Examples
    --------
    >>> from hydra_zen import builds, instantiate, just

    **Basic Usage**

    Instantiating a config that targets a class/type.

    >>> ConfDict = builds(dict, x=1)  # a targeted config
    >>> instantiate(ConfDict)  # calls `dict(x=1)`
    {'x': 1}

    Instantiating a config that targets a function.

    >>> def f(z): return z
    >>> ConfF = builds(f, z=22)  # a targeted config
    >>> instantiate(ConfF)  # calls `f(z=22)`
    22

    Providing a manual override, via ``instantiate(..., **kwargs)``

    >>> instantiate(ConfF, z='foo')  # calls `f(z='foo')`
    'foo'

    Recursive instantiation through nested configs.

    >>> inner = builds(dict, b="hi")
    >>> outer = builds(dict, a=inner)
    >>> instantiate(outer) # calls `dict(a=dict(b='hi))`
    {'a': {'b': 'hi'}}

    **Leveraging Variable Interpolation**

    Hydra provides a powerful language for absolute and relative
    interpolated variables among configs [4]_. Let's make a config
    where multiple fields reference the field ``name`` via absolute
    interpolation.

    >>> from hydra_zen import make_config
    >>> Conf = make_config("name", a="${name}", b=builds(dict, x="${name}"))

    Resolving the interpolation key: ``name``

    >>> instantiate(Conf, name="Jeff")
    {'a': 'Jeff', 'b': {'x': 'Jeff'}, 'name': 'Jeff'}

    **Runtime Data Validation via Hydra**

    >>> def g(x: float): return x  # note the annotation: float
    >>> Conf_g = builds(g, populate_full_signature=True)
    >>> instantiate(Conf_g, x=1.0)
    1.0

    Passing a non-float to ``x`` will produce a validation error upon instantiation

    >>> instantiate(Conf_g, x='hi')
    ValidationError: Value 'hi' could not be converted to Float
        full_key: x
        object_type=Builds_g

    Only a subset of primitive types are supported by Hydra's validation system [2]_.
    See :ref:`data-val` for more general data validation capabilities via hydra-zen.
    """
    return hydra_instantiate(config, *args, **kwargs)


def _apply_just(fn: F) -> F:
    @wraps(fn)
    def wrapper(cfg: Any, *args: Any, **kwargs: Any):
        if not is_dataclass(cfg):
            cfg = just(cfg)
        return fn(cfg, *args, **kwargs)

    return cast(F, wrapper)


@_apply_just
def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
    """
    Serialize a config as a yaml-formatted string.

    This is an alias of ``omegaconf.Omegaconf.to_yaml``.

    Parameters
    ----------
    cfg : Any
        A valid configuration object, supported either by Hydra or hydra-zen

    resolve : bool, optional (default=False)
        If `True`, interpolated fields in `cfg` will be resolved in the yaml.

    sort_keys : bool, optional (default=False)
        If `True`, the yaml's entries will alphabetically ordered.

    Returns
    -------
    yaml : str

    See Also
    --------
    save_as_yaml: Save a config to a yaml-format file.
    load_from_yaml: Load a config from a yaml-format file.

    Examples
    --------
    >>> from hydra_zen import builds, make_config, to_yaml

    **Basic usage**

    The yaml of a config with both an un-configured field
    and a configured field:

    >>> c1 = make_config("a", b=1)
    >>> print(to_yaml(c1))
    a: ???
    b: 1

    The yaml of a targeted config:

    >>> c2 = builds(dict, y=10)
    >>> print(to_yaml(c2))
    _target_: builtins.dict
    'y': 10

    hydra-zen's additional supported types can be specified as well

    >>> print(to_yaml(1+2j))
    real: 1.0
    imag: 2.0
    _target_: builtins.complex

    **Specifying resolve**

    The following is a config with interpolated fields.

    >>> c3 = make_config(a=builds(dict, b="${c}"), c=1)


    >>> print(to_yaml(c3, resolve=False))
    a:
      _target_: builtins.dict
      b: ${c}
    c: 1

    >>> print(to_yaml(c3, resolve=True))
    a:
      _target_: builtins.dict
      b: 1
    c: 1

    **Specifying sort_keys**

    >>> c4 = make_config("b", "a")  # field order: b then a

    >>> print(to_yaml(c4, sort_keys=False))
    b: ???
    a: ???

    >>> print(to_yaml(c4, sort_keys=True))
    a: ???
    b: ???
    """

    return OmegaConf.to_yaml(cfg=cfg, resolve=resolve, sort_keys=sort_keys)


@_apply_just
def save_as_yaml(
    config: Any, f: Union[str, pathlib.Path, IO[Any]], resolve: bool = False
) -> None:
    """
    Save a config to a yaml-format file

    This is an alias of ``omegaconf.Omegaconf.save`` [1]_.

    Parameters
    ----------
    config : Any
        A config object.

    f : str | pathlib.Path | IO[Any]
        The path of the file file, or a file object, to be written to.

    resolve : bool, optional (default=None)
        If ``True`` interpolations will be resolved in the config prior to serialization [2]_.
        See Examples section of `to_yaml` for details.

    See Also
    --------
    to_yaml: Serialize a config as a yaml-formatted string.
    load_from_yaml: Load a config from a yaml-format file.

    References
    ----------
    .. [1] https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#save-load-yaml-file
    .. [2] https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#variable-interpolation

    Examples
    --------
    >>> from hydra_zen import make_config, save_as_yaml, load_from_yaml

    **Basic usage**

    >>> Conf = make_config(a=1, b="foo")
    >>> save_as_yaml(Conf, "test.yaml")  # file written to: test.yaml
    >>> load_from_yaml("test.yaml")
    {'a': 1, 'b': 'foo'}
    """
    return OmegaConf.save(config=config, f=f, resolve=resolve)


def load_from_yaml(
    file_: Union[str, pathlib.Path, IO[Any]]
) -> Union[DictConfig, ListConfig]:
    """
    Load a config from a yaml-format file

    This is an alias of ``omegaconf.OmegaConf.load``.

    Parameters
    ----------
    file_ : str | pathlib.Path | IO[Any]
        The path to the yaml-formatted file, or the file object, that the config
        will be loaded from.

    Returns
    -------
    loaded_conf : DictConfig | ListConfig

    See Also
    --------
    save_as_yaml: Save a config to a yaml-format file.
    to_yaml: Serialize a config as a yaml-formatted string.

    References
    ----------
    .. [1] https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#save-load-yaml-file

    Examples
    --------
    >>> from hydra_zen import make_config, save_as_yaml, load_from_yaml

    **Basic usage**

    >>> Conf = make_config(a=1, b="foo")
    >>> save_as_yaml(Conf, "test.yaml")  # file written to: test.yaml
    >>> load_from_yaml("test.yaml")
    {'a': 1, 'b': 'foo'}
    """
    return OmegaConf.load(file_)
