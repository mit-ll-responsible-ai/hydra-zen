# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing import NamedTuple, Optional

import hydra
import omegaconf
from typing_extensions import Final


class Version(NamedTuple):
    major: int
    minor: int
    patch: Optional[int] = None


def get_version(ver_str: str) -> Version:
    assert ver_str.count(".") >= 2
    major, minor, *_ = (int(v) for v in ver_str.split("."))
    return Version(major=major, minor=minor)


OMEGACONF_VERSION: Final = get_version(omegaconf.__version__)
HYDRA_VERSION: Final = get_version(hydra.__version__)


# OmegaConf issue 830 describes a bug associated with structured configs
# composed via inheritance, where the child's attribute is a default-factory
# and the parent's corresponding attribute is not.
# We provide downstream workarounds until an upstream fix is released.
#
# Uncomment dynamic setting once OmegaConf merges fix:
# https://github.com/omry/omegaconf/pull/832
PATCH_OMEGACONF_830: Final = True  # OMEGACONF_VERSION <= Version(2, 1)

# Hydra's instantiate API now supports partial-instantiation, indicated
# by a `_partial_ = True` attribute.
# https://github.com/facebookresearch/hydra/pull/1905
HYDRA_SUPPORTS_PARTIAL: Final = Version(1, 1) < HYDRA_VERSION
