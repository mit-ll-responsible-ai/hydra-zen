# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from ._hydra_overloads import instantiate, to_yaml
from ._version import get_versions
from .structured_configs import builds, hydrated_dataclass, just, mutable_value

__version__ = get_versions()["version"]
del get_versions
