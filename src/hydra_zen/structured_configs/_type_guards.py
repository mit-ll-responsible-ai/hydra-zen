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
    ZEN_PARTIAL_TARGET_FIELD_NAME,
    ZEN_PROCESSING_LOCATION,
    ZEN_TARGET_FIELD_NAME,
)

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
    if not is_builds(x) or not hasattr(x, ZEN_TARGET_FIELD_NAME):
        return False
    attr = _get_target(x)
    if attr != ZEN_PROCESSING_LOCATION and attr is not zen_processing:
        return False
    return True


def is_partial_builds(x: Any) -> TypeGuard[PartialBuilds[Any]]:
    if is_builds(x):
        return (
            # check if partial'd config via Hydra
            HYDRA_SUPPORTS_PARTIAL
            and getattr(x, PARTIAL_FIELD_NAME, False) is True
        ) or (
            # check if partial'd config via `zen_processing`
            uses_zen_processing(x)
            and (getattr(x, ZEN_PARTIAL_TARGET_FIELD_NAME, False) is True)
        )
    return False
