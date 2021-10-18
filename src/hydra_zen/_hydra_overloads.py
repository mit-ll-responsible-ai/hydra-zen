# Copyright (c) 2021 Massachusetts Institute of Technology
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

import pathlib
from typing import IO, Any, Callable, Type, TypeVar, Union, overload

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .typing import Builds, Just, Partial, PartialBuilds
from .typing._implementations import _DataClass

__all__ = ["instantiate", "to_yaml", "save_as_yaml", "load_from_yaml", "MISSING"]


T = TypeVar("T")


@overload
def instantiate(
    config: Union[Just[T], Type[Just[T]]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[
        PartialBuilds[Callable[..., T]], Type[PartialBuilds[Callable[..., T]]]
    ],
    *args: Any,
    **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[PartialBuilds[Type[T]], Type[PartialBuilds[Type[T]]]],
    *args: Any,
    **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[Builds[Type[T]], Type[Builds[Type[T]]]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[Builds[Callable[..., T]], Type[Builds[Callable[..., T]]]],
    *args: Any,
    **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Union[ListConfig, DictConfig, _DataClass, Type[_DataClass]], *args, **kwargs
) -> Any:  # pragma: no cover
    ...


def instantiate(config: Any, *args, **kwargs) -> Any:
    """
    Instantiates the target of a targeted config.

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
    .. [1] https://hydra.cc/docs/next/advanced/instantiate_objects/overview/#recursive-instantiation
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


def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
    """
    Serialize a config as a yaml-formatted string.

    This is an alias of ``omegaconf.Omegaconf.to_yaml``.

    Parameters
    ----------
    cfg : Any
        A valid configuration object (e.g. DictConfig, ListConfig, dataclass).

    resolve : bool, optional (default=False)
        If `True`, interpolated fields in `cfg` will be resolved in the yaml.

    sort_keys : bool, optional (default=False)
        If `True`, the yaml's entries will alphabetically ordered.

    Returns
    -------
    yaml : str

    Examples
    --------
    >>> from hydra_zen import builds, to_yaml
    >>> cfg = builds(dict, a=builds(dict, b="${c}"), c=1)
    >>> print(to_yaml(cfg, resolve=True))
    _target_: builtins.dict
    a:
      _target_: builtins.dict
      b: 1
    c: 1
    """
    return OmegaConf.to_yaml(cfg=cfg, resolve=resolve, sort_keys=sort_keys)


def save_as_yaml(
    config: Any, f: Union[str, pathlib.Path, IO[Any]], resolve: bool = False
) -> None:
    """
    Save as configuration object to a yaml-format file

    This is an alias of ``omegaconf.Omegaconf.save`` [1]_.

    Parameters
    ----------
    config : Any
        A valid config object (e.g. a structured config produced by `builds`)

    f : str | pathlib.Path | IO[Any]]
        The target filename or file object

    resolve : bool, optional (default=None)
        If ``True`` interpolations will be resolved in the config prior to serialization [2]_.

    References
    ----------
    .. [1] https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#save-load-yaml-file
    .. [2] https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#variable-interpolation

    Examples
    --------
    >>> from hydra_zen import builds, save_as_yaml, load_from_yaml
    >>> conf = builds(dict, hello=10, goodbye=2)
    >>> save_as_yaml(conf, "test.yaml")  # file written to: test.yaml
    >>> load_from_yaml("test.yaml")
    {'_target_': 'builtins.dict', 'hello': 10, 'goodbye': 2}
    """
    return OmegaConf.save(config=config, f=f, resolve=resolve)


def load_from_yaml(
    file_: Union[str, pathlib.Path, IO[Any]]
) -> Union[DictConfig, ListConfig]:
    """
    Load as configuration object from a yaml-format file

    This is an alias of ``omegaconf.OmegaConf.load``.

    Parameters
    ----------
    file_ : str | pathlib.Path | IO[Any]
        A valid config object (e.g. a structured config produced by `builds`)

    Returns
    -------
    loaded_conf : DictConfig | ListConfig

    References
    ----------
    [1].. https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#save-load-yaml-file

    Examples
    --------
    >>> from hydra_zen import builds, save_as_yaml, load_from_yaml
    >>> conf = builds(dict, hello=10, goodbye=2)
    >>> save_as_yaml(conf, "test.yaml")  # file written to: test.yaml
    >>> load_from_yaml("test.yaml")
    {'_target_': 'builtins.dict', 'hello': 10, 'goodbye': 2}
    """
    return OmegaConf.load(file_)
