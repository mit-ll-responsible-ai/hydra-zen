# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
from typing import FrozenSet

from typing_extensions import Final

from hydra_zen._compatibility import HYDRA_SUPPORTS_PARTIAL
from hydra_zen.funcs import get_obj, zen_processing
from hydra_zen.structured_configs import _utils

# Hydra-specific fields
TARGET_FIELD_NAME: Final[str] = "_target_"
PARTIAL_FIELD_NAME: Final[str] = "_partial_"
RECURSIVE_FIELD_NAME: Final[str] = "_recursive_"
CONVERT_FIELD_NAME: Final[str] = "_convert_"
POS_ARG_FIELD_NAME: Final[str] = "_args_"
DEFAULTS_LIST_FIELD_NAME: Final[str] = "defaults"

_names = [
    TARGET_FIELD_NAME,
    RECURSIVE_FIELD_NAME,
    CONVERT_FIELD_NAME,
    POS_ARG_FIELD_NAME,
]

if HYDRA_SUPPORTS_PARTIAL:  # pragma: no cover
    _names.append(PARTIAL_FIELD_NAME)

HYDRA_FIELD_NAMES: FrozenSet[str] = frozenset(_names)

del _names

# hydra-zen-specific fields
ZEN_PROCESSING_LOCATION: Final[str] = _utils.get_obj_path(zen_processing)
GET_OBJ_LOCATION: Final[str] = _utils.get_obj_path(get_obj)
ZEN_TARGET_FIELD_NAME: Final[str] = "_zen_target"
ZEN_PARTIAL_FIELD_NAME: Final[str] = "_zen_partial"
META_FIELD_NAME: Final[str] = "_zen_exclude"
ZEN_WRAPPERS_FIELD_NAME: Final[str] = "_zen_wrappers"
JUST_FIELD_NAME: Final[str] = "path"
