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


@pytest.mark.parametrize(
    "target",
    [
        foo,
        partial(foo),
        builds(foo, populate_full_signature=True),
    ],
)
def test_manual(target):
    assert instantiate(
        builds(
            target,
            populate_full_signature=True,
            zen_dataclass={"target": "builtins.dict"},
        )
    ) == dict(x=1, y=2)
