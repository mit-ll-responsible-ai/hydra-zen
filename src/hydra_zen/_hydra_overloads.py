# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

"""
Provides annotation overloads for various hydra functions, using the types defined in `hydra_utils.typing`.
This enables tools like IDEs to be more incisive during static analysis and to provide users with additional
context about their code.

E.g.

.. code::

   from hydra_utils import builds, instantiate
   DictConfig = builds(dict, a=1, b=2)  # type: Builds[Type[dict]]

   # static analysis tools can provide useful type information
   # about the object that is instantiated from the config
   out = instantiate(DictConfig)  # type: dict

"""

import pathlib
from typing import IO, Any, Callable, Type, TypeVar, Union, overload

from hydra.utils import instantiate as hydra_instantiate
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf

from .typing import Builds, Just, Partial, PartialBuilds

__all__ = ["instantiate", "to_yaml", "save_as_yaml", "load_from_yaml", "MISSING"]


T = TypeVar("T")


@overload
def instantiate(
    config: Type[Just[T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(config: Just[T], *args: Any, **kwargs: Any) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: PartialBuilds[Callable[..., T]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Type[PartialBuilds[Callable[..., T]]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Type[PartialBuilds[Type[T]]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: PartialBuilds[Type[T]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Type[Builds[Type[T]]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Builds[Type[T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Type[Builds[Callable[..., T]]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Builds[Callable[..., T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


def instantiate(config: Any, *args, **kwargs) -> Any:
    """
    Instantiates the object specified in the ``_target_`` field of ``config``.

    Calls `hydra.utils.instantiate(config, *args, **kwargs)`

    This functions is identical to `hydra.utils.instantiate`, but it provides
    useful static type information by leveraging the types defined in `hydra_utils.typing`.

    Parameters
    ----------
    config : Instantiable[Type[T]]
        The config object (dict or structured config) to be instantiated.

    *args: Any
        Positional parameters pass-through.

    **kwargs : Any
        Named parameters to override parameters in the config object.
        Parameters not present in the config objects are being passed as is to the target.
           IMPORTANT: dataclasses instances in kwargs are interpreted as config
                      and cannot be used as passthrough

    Returns
    -------
    instantiated : T
        The config, instantiated with the arguments specified in the config, and by
        *args and **kwargs.

    Examples
    --------
    >>> from hydra_zen import instantiate, builds
    >>> config = builds(dict, a=1, b=2)  # type: Type[Builds[Type[dict]]]
    >>> instantiate(config, c=3)  # static analysis can deduce that the result type is `dict`
    {'a': 1, 'b': 2, 'c': 3}

    >>> config = builds(list)  # type: Type[Builds[Type[list]]]
    >>> instantiate(config, (1, 2, 3))  # static analysis can deduce that the result type is `list`
    [1, 2, 3]"""
    return hydra_instantiate(config, *args, **kwargs)


def to_yaml(cfg: Any, *, resolve: bool = False, sort_keys: bool = False) -> str:
    """
    Returns a yaml-formatted string representation of ``cfg``.

    This is an alias of ``omegaconf.Omegaconf.to_yaml`` [1]_

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
    >>> cfg = builds(dict, a=builds(dict, b="${a}"))  # structured config of nested dictionaries
    >>> print(to_yaml(cfg, resolve=True))
    _target_: builtins.dict
    _recursive_: true
    _convert_: none
    a:
      _target_: builtins.dict
      _recursive_: true
      _convert_: none
      b: 1
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
    [1].. https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#save-load-yaml-file
    [2].. https://omegaconf.readthedocs.io/en/2.0_branch/usage.html#variable-interpolation

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

    This is an alias of ``omegaconf.OmegaConf.load`` [1]_.

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
