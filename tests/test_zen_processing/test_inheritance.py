from dataclasses import dataclass
from functools import partial

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, instantiate, to_yaml
from hydra_zen._compatibility import HYDRA_SUPPORTS_PARTIAL
from hydra_zen.structured_configs._globals import (
    PARTIAL_FIELD_NAME,
    ZEN_PROCESSING_LOCATION,
)
from hydra_zen.structured_configs._type_guards import uses_zen_processing
from hydra_zen.typing import HydraPartialBuilds, ZenPartialBuilds
from tests import sorted_yaml


def wrapper1(func):
    return func


def wrapper2(func):
    return func


@given(
    args=st.dictionaries(st.sampled_from("abc"), st.integers(-5, 5)),
    meta=st.none() | st.dictionaries(st.sampled_from("def"), st.integers(-5, 5)),
    wrappers=st.lists(st.sampled_from([None, wrapper1, wrapper2])),
    meta_via_parent=st.booleans(),
    wrappers_via_parent=st.booleans(),
)
def test_inherited_zen_processing(
    args, meta, wrappers, meta_via_parent, wrappers_via_parent
):
    parent_kwargs = {}
    child_kwargs = {}

    target = parent_kwargs if meta_via_parent else child_kwargs
    target["zen_meta"] = meta

    target = parent_kwargs if wrappers_via_parent else child_kwargs
    target["zen_wrappers"] = wrappers

    parent = builds(dict, **args, **parent_kwargs)
    child = builds(dict, **child_kwargs, builds_bases=(parent,))

    direct = builds(dict, **args, zen_meta=meta, zen_wrappers=wrappers)

    assert sorted_yaml(direct) == sorted_yaml(child)
    assert instantiate(direct) == instantiate(child)


def yaml_lines(conf):
    return set(to_yaml(conf).splitlines())


@given(
    args=st.dictionaries(st.sampled_from("abc"), st.integers(-5, 5)),
    meta=st.none() | st.dictionaries(st.sampled_from("def"), st.integers(-5, 5)),
    wrappers=st.lists(st.sampled_from([None, wrapper1, wrapper2])),
)
def test_partial_with_inherited_zen_processing(args, meta, wrappers):
    parent = builds(dict, **args, zen_meta=meta, zen_wrappers=wrappers)
    Child = builds(dict, zen_partial=True, builds_bases=(parent,))

    child = Child()

    if HYDRA_SUPPORTS_PARTIAL or uses_zen_processing(parent):
        # ensure meta fields are retained by child
        assert yaml_lines(parent) <= yaml_lines(child)

    if uses_zen_processing(parent) or not HYDRA_SUPPORTS_PARTIAL:
        assert uses_zen_processing(child)
        assert isinstance(child, ZenPartialBuilds)
        assert not isinstance(child, HydraPartialBuilds)
    else:
        assert not uses_zen_processing(child)
        assert isinstance(child, HydraPartialBuilds)

    out = instantiate(child)
    assert isinstance(out, partial)
    assert out() == args


@dataclass
class ZenPartialFalse:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_target: str = "builtins.dict"
    _zen_partial: bool = False
    a: int = 1


@dataclass
class ZenPartialTrue:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_target: str = "builtins.dict"
    _zen_partial: bool = True
    a: int = 1


@dataclass
class HydraPartialFalse:
    _target_: str = "builtins.dict"
    _partial_: bool = False
    a: int = 1


@dataclass
class HydraPartialTrue:
    _target_: str = "builtins.dict"
    _partial_: bool = True
    a: int = 1


@pytest.mark.parametrize("Parent", [ZenPartialFalse, HydraPartialFalse])
def test_inherited_partial_false_fields(Parent):
    if not HYDRA_SUPPORTS_PARTIAL and Parent is HydraPartialFalse:
        pytest.mark.skip("Hydra version doesn't support instantiate API")

    Conf = builds(dict, builds_bases=(Parent,))  # should be OK
    assert instantiate(Conf) == instantiate(Parent)


def test_inherit_hydra_partial_with_zen_processing_raises():
    with pytest.raises(TypeError):
        builds(
            dict, zen_meta=dict(a=1), zen_partial=True, builds_bases=(HydraPartialTrue,)
        )


@pytest.mark.parametrize("Parent", [ZenPartialTrue, HydraPartialTrue])
def test_zen_partial_false_raises_for_partiald_parents(Parent):
    with pytest.raises(TypeError):
        builds(dict, zen_partial=False, builds_bases=(Parent,))


@pytest.mark.parametrize(
    "Parent", [ZenPartialFalse, HydraPartialFalse, ZenPartialTrue, HydraPartialTrue]
)
def test_zen_partial_true_holds_for_all_inheritance(Parent):
    def make_conf():
        return builds(dict, b=2, zen_partial=True, builds_bases=(Parent,))

    if not HYDRA_SUPPORTS_PARTIAL and hasattr(Parent, PARTIAL_FIELD_NAME):
        with pytest.raises(TypeError):
            make_conf()
        return

    Conf = make_conf()

    out = instantiate(Conf)
    assert isinstance(out, partial)
    assert out() == {"a": 1, "b": 2}
