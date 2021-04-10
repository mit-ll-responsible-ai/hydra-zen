# Copyright (c) 2021 Massachusetts Institute of Technology

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

from typing import Any, Callable, Type, TypeVar, overload

from hydra.utils import instantiate as hydra_instantiate

from .typing import Builds, Just, Partial, PartialBuilds

T = TypeVar("T")

Callable_T = TypeVar("Callable_T", bound=Callable)


@overload
def instantiate(config: Just[T], *args: Any, **kwargs: Any) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: PartialBuilds[Type[T]], *args: Any, **kwargs: Any
) -> Partial[T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: PartialBuilds[Callable_T], *args: Any, **kwargs: Any
) -> Partial[Callable_T]:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Builds[Type[T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


@overload
def instantiate(
    config: Builds[Callable[..., T]], *args: Any, **kwargs: Any
) -> T:  # pragma: no cover
    ...


def instantiate(config: Any, *args, **kwargs) -> Any:
    """Calls `hydra.utils.instantiate(config, *args, **kwargs)`

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
    >>> config = builds(dict, a=1, b=2)  # type: Builds[Type[dict]]
    >>> instantiate(config, c=3)  # static analysis can deduce that the result type is `dict`
    {'a': 1, 'b': 2, 'c': 3}

    >>> config = builds(list)  # type: Builds[Type[list]]
    >>> instantiate(config, (1, 2, 3))  # static analysis can deduce that the result type is `list`
    [1, 2, 3]"""
    return hydra_instantiate(config, *args, **kwargs)
