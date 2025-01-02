# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: strict
from typing import Final

from hydra_zen.funcs import zen_processing

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
    PARTIAL_FIELD_NAME,
]


HYDRA_FIELD_NAMES: frozenset[str] = frozenset(_names)

del _names

# hydra-zen-specific fields
ZEN_PROCESSING_LOCATION: Final[str] = ".".join(
    [zen_processing.__module__, zen_processing.__name__]
)
ZEN_TARGET_FIELD_NAME: Final[str] = "_zen_target"
ZEN_PARTIAL_FIELD_NAME: Final[str] = "_zen_partial"
META_FIELD_NAME: Final[str] = "_zen_exclude"
ZEN_WRAPPERS_FIELD_NAME: Final[str] = "_zen_wrappers"
JUST_FIELD_NAME: Final[str] = "path"
