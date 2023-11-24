# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import pytest

from hydra_zen import builds, instantiate, make_custom_builds_fn


@pytest.mark.parametrize("bad_exclude", ["x", [["x"]]])
def test_validate_exclude(bad_exclude):
    with pytest.raises(TypeError, match="must only contain"):
        builds(dict, zen_exclude=bad_exclude)


def foo(x=1, _y=2, _z=3):
    return "apple"


@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("custom_builds", [True, False])
@pytest.mark.parametrize(
    "exclude",
    [
        ["_y", "_z"],
        lambda x: x.startswith("_"),
        [1, 2],
        ["_y", -1],
        ["_y", 2],
    ],
)
def test_exclude_named(partial: bool, custom_builds: bool, exclude):
    if custom_builds:
        b = make_custom_builds_fn(
            populate_full_signature=True, zen_partial=partial, zen_exclude=exclude
        )
        conf = b(foo)()
    else:
        conf = builds(
            foo, populate_full_signature=True, zen_partial=partial, zen_exclude=exclude
        )()
    assert not hasattr(conf, "_y")
    assert not hasattr(conf, "_z")
    assert conf.x == 1
    if not partial:
        assert instantiate(conf) == "apple"
    else:
        assert instantiate(conf)() == "apple"  # type: ignore
