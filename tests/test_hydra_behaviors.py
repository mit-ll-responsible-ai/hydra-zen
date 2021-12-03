# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field, is_dataclass
from typing import Any, List, Tuple

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from omegaconf import MISSING, DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ValidationError

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    mutable_value,
)
from hydra_zen.errors import HydraZenValidationError
from hydra_zen.structured_configs._utils import PATCH_OMEGACONF_830, get_obj_path


def f_three_vars(x, y, z):
    return x, y, z


def wrapper(obj):
    return obj


@given(
    convert=st.sampled_from(["none", "all", "partial"]),
    recursive=st.booleans(),
    sig=st.booleans(),
    partial=st.booleans(),
    meta=st.none() | st.just(dict(meta=1)),
    wrappers=st.none() | st.just(wrapper) | st.lists(st.just(wrapper)),
    kwargs=st.dictionaries(keys=st.sampled_from(["x", "y", "z"]), values=st.floats()),
    name=st.none() | st.sampled_from(["NameA", "NameB"]),
)
def test_builds_sets_hydra_params(
    convert, recursive, sig, partial, name, meta, wrappers, kwargs
):
    out = builds(
        f_three_vars,
        hydra_convert=convert,
        hydra_recursive=recursive,
        populate_full_signature=sig,
        zen_partial=partial,
        zen_meta=meta,
        zen_wrappers=wrappers,
        dataclass_name=name,
        **kwargs,
    )

    assert out._convert_ == convert
    assert out._recursive_ == recursive
    if name is not None:
        assert out.__name__ == name
    else:
        assert "Builds_f_three_vars" in out.__name__


def f_for_convert(**kwargs):
    return kwargs


class NotSet:
    pass


@dataclass
class A:
    x: Any = (1, 2)


@pytest.mark.parametrize("via_hydrated_dataclass", [False, True])
@pytest.mark.parametrize(
    "convert, expected_types",
    [
        # "Passed objects are DictConfig and ListConfig"
        ("none", (DictConfig, ListConfig, DictConfig, int)),
        # default should be same as 'none'
        (NotSet, (DictConfig, ListConfig, DictConfig, int)),
        # "Passed objects are converted to dict and list, with
        #  the exception of Structured Configs (and their fields)."
        ("partial", (dict, list, DictConfig, int)),
        # "Passed objects are dicts, lists and primitives without
        # a trace of OmegaConf containers"
        ("all", (dict, list, dict, int)),
    ],
)
def test_hydra_convert(
    convert: str, expected_types: List[type], via_hydrated_dataclass: bool
):
    """Tests that the `hydra_convert` parameter produces the expected/documented
    behavior in hydra."""

    kwargs = dict(hydra_convert=convert) if convert is not NotSet else {}

    if not via_hydrated_dataclass:

        out = instantiate(
            builds(
                f_for_convert,
                a_dict=dict(x=1),
                a_list=[1, 2],
                a_struct_config=A,
                an_int=1,
                **kwargs,
            )
        )
    else:

        @hydrated_dataclass(f_for_convert, **kwargs)
        class MyConf:
            a_dict: Any = mutable_value(dict(x=1))
            a_list: Any = mutable_value([1, 2])
            a_struct_config: Any = A
            an_int: int = 1

        out = instantiate(MyConf)

    actual_types = tuple(
        type(out[i]) for i in ("a_dict", "a_list", "a_struct_config", "an_int")
    )
    assert actual_types == expected_types

    if convert in ("none", "partial", NotSet):
        expected_field_type = ListConfig
    else:
        expected_field_type = list

    assert type(out["a_struct_config"]["x"]) is expected_field_type


def f_for_recursive(**kwargs):
    return kwargs


@pytest.mark.parametrize("via_hydrated_dataclass", [False, True])
@pytest.mark.parametrize("recursive", [False, True, NotSet])
def test_recursive(recursive: bool, via_hydrated_dataclass: bool):
    target_path = get_obj_path(f_for_recursive)

    kwargs = dict(hydra_recursive=recursive) if recursive is not NotSet else {}

    if not via_hydrated_dataclass:
        out = instantiate(
            builds(
                f_for_recursive,
                x=builds(f_for_recursive, y=1),
                **kwargs,
            )
        )

    else:

        @hydrated_dataclass(f_for_recursive)
        class B:
            y: int = 1

        @hydrated_dataclass(f_for_recursive, **kwargs)
        class A:
            x: Any = B

        out = instantiate(A)

    if recursive is NotSet or recursive:
        assert out == {"x": {"y": 1}}
    else:
        assert out == {"x": {"_target_": target_path, "y": 1}}


def f(x: Tuple[int]):
    return x


class C:
    def __init__(self, x: int):
        self.x = x


def g(x: C):
    return x


def g2(x: List[C]):
    return x


def test_type_checking():
    conf = builds(f, populate_full_signature=True)(x=("hi",))
    with pytest.raises(ValidationError):
        instantiate(conf)

    # should be ok
    instantiate(builds(f, populate_full_signature=True)(x=(1,)))

    # nested configs should get validated too
    with pytest.raises(ValidationError):
        instantiate(builds(g, x=builds(C, x="hi")))  # Invalid: `C.x` must be int

    # should be ok
    instantiate(builds(g, x=builds(C, x=1)))

    conf_C = builds(C, x=1)

    # should be ok
    instantiate(builds(g2, x=[conf_C, conf_C]))


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
        # ensure we only case to dataclass when necessary
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
