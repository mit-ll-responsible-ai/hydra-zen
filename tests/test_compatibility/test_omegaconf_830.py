# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field, is_dataclass
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from omegaconf import MISSING, OmegaConf

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    mutable_value,
)
from hydra_zen._compatibility import PATCH_OMEGACONF_830
from hydra_zen.errors import HydraZenValidationError


def test_PATCH_OMEGACONF_830_is_set_properly():
    # test that PATCH_OMEGACONF_830 is True only if local version
    # of omegaconf has known bug
    assert isinstance(PATCH_OMEGACONF_830, bool)

    @dataclass
    class BasicConf:
        setup: Any = 1

    @dataclass
    class Config(BasicConf):
        setup: Any = field(default_factory=lambda: list(["hi"]))

    conf = OmegaConf.structured(Config)
    if PATCH_OMEGACONF_830:
        assert conf.setup == 1
    else:
        # local version of omegaconf should have correct behavior
        assert conf.setup == ["hi"]


@dataclass
class A_inheritance:
    x: Any
    y: Any = 1


valid_defaults = (
    st.none()
    | st.booleans()
    | st.text(alphabet="abcde")
    | st.integers(-3, 3)
    | st.lists(st.integers(-2, 2))
    | st.fixed_dictionaries({"a": st.integers(-2, 2)})
)


def via_hydrated(x, Parent):
    z = x if not isinstance(x, (list, dict)) else mutable_value(x)

    @hydrated_dataclass(A_inheritance)
    class Conf(Parent):
        x: Any = z

    return Conf


def mutable_if_needed(x):
    if isinstance(x, (dict, list)):
        return mutable_value(x)
    return x


@pytest.mark.parametrize(
    "config_maker",
    [
        lambda x, Parent: make_config(x=x, bases=(Parent,)),
        lambda x, Parent: make_config(x=ZenField(Any, x), bases=(Parent,)),
        lambda x, Parent: make_config(ZenField(Any, x, "x"), bases=(Parent,)),
        lambda x, Parent: builds(A_inheritance, x=x, builds_bases=(Parent,)),
        # TODO: add case where x is populated via pop-full-sig
        # we currently support specifing fields-as-args in builds
        lambda x, Parent: builds(
            A_inheritance, x=mutable_if_needed(x), builds_bases=(Parent,)
        ),
        via_hydrated,
    ],
)
@given(
    parent_field_name=st.sampled_from(["x", "y"]),
    parent_default=valid_defaults,
    child_default=valid_defaults,
)
def test_known_inheritance_issues_in_omegaconf_are_circumvented(
    parent_field_name, parent_default, child_default, config_maker
):
    # Exercises omegaconf bug documented in https://github.com/omry/omegaconf/issues/830
    # Should pass either because:
    #  - the test env is running a new version of omegaconf, which has patched this
    #  - hydra-zen is providing a workaround
    if PATCH_OMEGACONF_830 and config_maker is via_hydrated:
        pytest.skip("hydrated_dataclass cannot support patched workaround")

    assume(parent_field_name != "y" or parent_default is not MISSING)

    Parent = make_config(**{parent_field_name: parent_default})
    Child = config_maker(x=child_default, Parent=Parent)

    if (
        PATCH_OMEGACONF_830
        and isinstance(child_default, (list, dict))
        and not isinstance(parent_default, (list, dict))
        and parent_field_name == "x"  # parent field overlaps
    ):
        # ensure we only cast to dataclass when necessary
        assert is_dataclass(Child.x)
    else:
        assert Child().x == child_default

    Obj = instantiate(Child)
    assert Obj.x == child_default

    if parent_field_name == "y":
        assert Obj.y == parent_default


@pytest.mark.skipif(
    not PATCH_OMEGACONF_830, reason="issue has been patched by omegaconf"
)
def test_hydrated_dataclass_raises_on_omegaconf_inheritance_issue():

    Parent = make_config(x=1)

    with pytest.raises(HydraZenValidationError):

        @hydrated_dataclass(A_inheritance)
        class Conf(Parent):
            x: Any = mutable_value([1, 2])
