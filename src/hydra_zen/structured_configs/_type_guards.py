# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
from functools import partial
from typing import Any

from typing_extensions import TypeGuard

from hydra_zen._compatibility import HYDRA_SUPPORTS_PARTIAL
from hydra_zen.funcs import get_obj, zen_processing
from hydra_zen.typing import Builds, Just, PartialBuilds

from ._globals import (
    JUST_FIELD_NAME,
    PARTIAL_FIELD_NAME,
    TARGET_FIELD_NAME,
    ZEN_PARTIAL_FIELD_NAME,
    ZEN_PROCESSING_LOCATION,
    ZEN_TARGET_FIELD_NAME,
)

__all__ = ["is_partial_builds", "uses_zen_processing"]

# We need to check if things are Builds, Just, PartialBuilds to a higher
# fidelity than is provided by `isinstance(..., <Protocol>)`. I.e. we want to
# check that the desired attributes *and* that their values match those of the
# protocols. Failing to heed this would, for example, lead to any `Builds` that
# happens to have a `path` attribute to be treated as `Just` in `get_target`.
#
# The following functions perform these desired checks. Note that they do not
# require that the provided object be a dataclass; this enables compatibility
# with omegaconf containers.
#
# These are not part of the public API for now, but they may be in the future.


def _get_target(x: Any):
    return getattr(x, TARGET_FIELD_NAME)


def is_builds(x: Any) -> TypeGuard[Builds[Any]]:
    return hasattr(x, TARGET_FIELD_NAME)


def is_just(x: Any) -> TypeGuard[Just[Any]]:
    if is_builds(x) and hasattr(x, JUST_FIELD_NAME):
        attr = _get_target(x)
        if attr == _get_target(Just) or attr is get_obj:
            return True
        else:
            # ensures we conver this branch in tests
            return False
    return False


def is_old_partial_builds(x: Any) -> bool:  # pragma: no cover
    # We don't care about coverage here.
    # This will only be used in `get_target` and we'll be sure to cover that branch
    if is_builds(x) and hasattr(x, "_partial_target_"):
        attr = _get_target(x)
        if (attr == "hydra_zen.funcs.partial" or attr is partial) and is_just(
            getattr(x, "_partial_target_")
        ):
            return True
        else:
            # ensures we cover this branch in tests
            return False
    return False


def uses_zen_processing(x: Any) -> TypeGuard[Builds[Any]]:
    """Returns `True` if the input is a targeted structured config that relies on
    zen-processing features during its instantiation process. See notes for more details

    Parameters
    ----------
    x : Any

    Returns
    -------
    uses_zen : bool

    Notes
    -----
    In order to support zen :ref:`meta-fields <meta-field>` and
    :ref:`zen wrappers <zen-wrapper>`, hydra-zen redirects Hydra to an intermediary
    function – `hydra_zen.funcs.zen_processing` – during instantiation; i.e.
    `zen_processing` is made to be the `_target_` of the config and `_zen_target`
    indicates the object that is ultimately being configured for instantiation.

    Examples
    --------
    >>> from hydra_zen import builds, uses_zen_processing, to_yaml
    >>> ConfA = builds(dict, a=1)
    >>> ConfB = builds(dict, a=1, zen_partial=True)
    >>> ConfC = builds(dict, a=1, zen_wrappers=lambda x: x)
    >>> ConfD = builds(dict, a=1, zen_meta=dict(hidden_field=None))
    >>> ConfE = builds(dict, a=1, zen_meta=dict(hidden_field=None), zen_partial=True)
    >>> uses_zen_processing(ConfA)
    False
    >>> uses_zen_processing(ConfB)
    False
    >>> uses_zen_processing(ConfC)
    True
    >>> uses_zen_processing(ConfD)
    True
    >>> uses_zen_processing(ConfE)
    True

    Demonstrating the indirection that is used to facilitate zen-processing features.

    >>> print(to_yaml(ConfE))
    _target_: hydra_zen.funcs.zen_processing
    _zen_target: builtins.dict
    _zen_partial: true
    _zen_exclude:
    - hidden_field
    a: 1
    hidden_field: null
    """
    if not is_builds(x) or not hasattr(x, ZEN_TARGET_FIELD_NAME):
        return False
    attr = _get_target(x)
    if attr != ZEN_PROCESSING_LOCATION and attr is not zen_processing:
        return False
    return True


def is_partial_builds(x: Any) -> TypeGuard[PartialBuilds[Any]]:
    """
    Returns `True` if the input is a targeted structured config that entails partial
    instantiation, either via `_partial_=True` [1]_ or via `_zen_partial=True`.

    Parameters
    ----------
    x : Any

    Returns
    -------
    is_partial_config : bool

    References
    ----------
    .. [1] https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation

    See Also
    --------
    uses_zen_processing

    Examples
    --------
    >>> from hydra_zen import is_partial_builds

    An example involving a basic structured config

    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class A:
    ...     _target_ : str = 'builtins.int'
    ...     _partial_ : bool = True
    >>> is_partial_builds(A)
    True
    >>> is_partial_builds(A(_partial_=False))
    False

    An example of a config that leverages partial instantiation via zen-processing

    >>> from hydra_zen import builds, uses_zen_processing, instantiate
    >>> Conf = builds(int, 0, zen_partial=True, zen_meta=dict(a=1))
    >>> hasattr(Conf, "_partial_")
    False
    >>> uses_zen_processing(Conf)
    True
    >>> is_partial_builds(Conf)
    True
    >>> instantiate(Conf)
    functools.partial(<class 'int'>, 0)
    """
    if is_builds(x):
        return (
            # check if partial'd config via Hydra
            HYDRA_SUPPORTS_PARTIAL
            and getattr(x, PARTIAL_FIELD_NAME, False) is True
        ) or (
            # check if partial'd config via `zen_processing`
            uses_zen_processing(x)
            and (getattr(x, ZEN_PARTIAL_FIELD_NAME, False) is True)
        )
    return False
