# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys

assert sys.version_info > (3, 9)

from dataclasses import KW_ONLY, dataclass

import pytest
from omegaconf import OmegaConf

from hydra_zen import builds, get_target, instantiate, just
from hydra_zen.structured_configs._type_guards import safe_getattr
from hydra_zen.typing import DataclassOptions as Dc

sl = Dc(slots=True)


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


@pytest.mark.parametrize(
    "obj,field,expected",
    [
        (builds(int, x=2, zen_dataclass=Dc(slots=True)), "x", 2),
        (builds(int, zen_dataclass=Dc(slots=True)), "_target_", "builtins.int"),
        (
            builds(int, zen_partial=True, zen_dataclass=Dc(slots=True)),
            "_partial_",
            True,
        ),
    ],
)
def test_safe_getattr_with_slots(obj, field, expected):
    assert safe_getattr(obj, field) == expected
