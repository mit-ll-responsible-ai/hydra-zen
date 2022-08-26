# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys

assert sys.version_info > (3, 9)

from dataclasses import KW_ONLY, dataclass
from typing import Any, Dict, List, Tuple

import pytest
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, just


@dataclass
class Point:
    x: float
    _: KW_ONLY
    y: float
    z: float


def test_just_on_dataclass_w_kwonly_field():
    pt = Point(0, y=1.5, z=2.0)
    assert instantiate(just(pt)) == pt


def test_builds_on_dataclass_w_kwonly_field():
    pt = Point(0, y=1.5, z=2.0)
    Conf = builds(Point, populate_full_signature=True)
    assert instantiate(Conf(0, y=1.5, z=2.0)) == pt
