# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

import pytest

from hydra_zen import builds, instantiate, make_custom_builds_fn
from hydra_zen.structured_configs._utils import get_obj_path
from hydra_zen.typing import ZenConvert

fbuilds = make_custom_builds_fn(populate_full_signature=True)


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
    # In A, `_target_` is an init=True field, which
    # we will pick up in pop-sig; but _target_ is
    @dataclass
    class A:
        x: int
        y: int
        _target_: str = get_obj_path(foo)

    c = builds(A, x=3, y=4, populate_full_signature=True)
    assert c._target_ == A._target_
    assert instantiate(c) == (3, 4)


def test_store_hydrated():
    from hydra_zen import ZenStore, hydrated_dataclass, instantiate

    store = ZenStore()

    @store(name="a")
    @store(name="b", y=-1)
    @store(name="c", x=-1, y=2)
    @hydrated_dataclass(
        foo,
        zen_partial=True,  # <- note
    )
    class A:
        x: int
        y: int

    assert instantiate(A)(x=3, y=4) == (3, 4)
    assert instantiate(store[None, "a"])(x=10, y=22) == (10, 22)
    assert instantiate(store[None, "b"])(x=2) == (2, -1)
    assert instantiate(store[None, "c"])() == (-1, 2)
