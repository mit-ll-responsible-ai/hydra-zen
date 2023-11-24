# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from functools import partial

import pytest

from hydra_zen import builds, instantiate


@pytest.mark.parametrize(
    "target_path",
    [
        int,
        1,
        ["a"],
        "not a path",
    ],
)
def test_validation(target_path):
    with pytest.raises(TypeError, match="dataclass option `target`"):
        builds(int, zen_dataclass={"target": target_path})


def foo(x=1, y=2):
    raise AssertionError("I should not get called")


def passthrough(x):
    return x


@pytest.mark.parametrize(
    "target",
    [
        foo,
        partial(foo),
        builds(foo, populate_full_signature=True),
    ],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"x": 1},
        # overrides default with different value
        pytest.param({"x": 2}, marks=pytest.mark.xfail),
        # exercise zen_processing branch
        {"zen_wrappers": passthrough},
    ],
)
def test_manual(target, kwargs):
    out = instantiate(
        builds(
            target,
            populate_full_signature=True,
            zen_dataclass={"target": "builtins.dict"},
            **kwargs,
        )
    )
    assert isinstance(out, dict)
    assert out == dict(x=1, y=2)
