# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

import pytest

from hydra_zen import builds, instantiate
from hydra_zen.structured_configs._utils import get_obj_path
from hydra_zen.typing import ZenConvert


def test_no_flat_target():
    out = builds(builds(int), zen_convert=ZenConvert(flat_target=False))
    assert out._target_.startswith("types.Builds")
    with pytest.raises(Exception):
        instantiate(out)


@pytest.mark.parametrize("options", [ZenConvert(flat_target=True), ZenConvert()])
def test_flat_target(options: ZenConvert):
    out = builds(builds(int), zen_convert=options)
    assert instantiate(out) == int()


def foo(x: int, y: int):
    return (x, y)


def test_flat_target_manual_config():
    @dataclass
    class A:
        _target_: str = get_obj_path(foo)
        x: int = 1
        y: int = 2
