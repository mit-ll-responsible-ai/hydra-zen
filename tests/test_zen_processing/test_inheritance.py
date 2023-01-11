# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from functools import partial
from typing import Tuple, Type

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, instantiate, make_config, to_yaml
from hydra_zen._compatibility import HYDRA_SUPPORTS_PARTIAL
from hydra_zen.structured_configs._globals import (
    PARTIAL_FIELD_NAME,
    ZEN_PARTIAL_FIELD_NAME,
    ZEN_PROCESSING_LOCATION,
)
from hydra_zen.structured_configs._type_guards import (
    is_partial_builds,
    uses_zen_processing,
)
from hydra_zen.typing import HydraPartialBuilds, ZenPartialBuilds
from hydra_zen.typing._implementations import DataClass_, Partial
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
class HydraNoPartial:
    _target_: str = "builtins.dict"
    a: int = 1


@dataclass
class ZenNoPartial:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_target: str = "builtins.dict"
    a: int = 1


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


@dataclass
class ZenFalseHydraTrue:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_partial: bool = False
    _partial_: bool = True
    _zen_target: str = "builtins.dict"
    a: int = 1


@dataclass
class ZenTrueHydraFalse:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_partial: bool = True
    _partial_: bool = False
    _zen_target: str = "builtins.dict"
    a: int = 1


@dataclass
class ZenTrueHydraTrue:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_partial: bool = True
    _partial_: bool = True
    _zen_target: str = "builtins.dict"
    a: int = 1


@dataclass
class ZenFalseHydraFalse:
    _target_: str = ZEN_PROCESSING_LOCATION
    _zen_partial: bool = False
    _partial_: bool = False
    _zen_target: str = "builtins.dict"
    a: int = 1


parents_strat = st.lists(
    st.sampled_from(
        [
            HydraNoPartial,
            ZenNoPartial,
            HydraPartialTrue,
            HydraPartialFalse,
            ZenPartialTrue,
            ZenPartialFalse,
            ZenTrueHydraFalse,
            ZenFalseHydraTrue,
            ZenTrueHydraTrue,
            ZenFalseHydraFalse,
        ]
    ),
    unique=True,
).map(tuple)


@given(
    child_partial=st.none() | st.booleans(),
    zen_meta=st.sampled_from([{}, dict(meta=True)]),
    parents=parents_strat,
)
def test_partial_via_inheritance(
    child_partial: bool, zen_meta: dict, parents: Tuple[Type[DataClass_], ...]
):

    expected_out = dict(child_field=2)
    if parents:
        expected_out["a"] = 1

    parent_partial = None
    if child_partial is None:
        # First parent with partial explicit partial field dictates inheritance
        for p in parents:
            if HYDRA_SUPPORTS_PARTIAL and hasattr(p, PARTIAL_FIELD_NAME):
                parent_partial = getattr(p, PARTIAL_FIELD_NAME)

            if hasattr(p, ZEN_PARTIAL_FIELD_NAME):
                parent_partial = parent_partial or getattr(p, ZEN_PARTIAL_FIELD_NAME)

            if parent_partial is not None:
                break

    Conf = builds(
        dict,
        child_field=2,
        zen_partial=child_partial,
        zen_meta=zen_meta,
        builds_bases=parents,
    )

    expected_partial = bool(
        child_partial if child_partial is not None else parent_partial
    )

    assert is_partial_builds(Conf) is expected_partial

    if hasattr(Conf, PARTIAL_FIELD_NAME) and not HYDRA_SUPPORTS_PARTIAL:
        expected_out[PARTIAL_FIELD_NAME] = getattr(Conf, PARTIAL_FIELD_NAME)

    # check actual instantiation behavior
    out = instantiate(Conf)
    if expected_partial:
        out: Partial[dict]
        assert out() == expected_out
    else:
        assert isinstance(out, dict)
        assert out == expected_out


@given(
    child_partial=st.none() | st.booleans(),
    zen_meta=st.sampled_from([{}, dict(meta=True)]),
    parents=parents_strat,
)
def test_instantiation_never_produces_partiald_zen_processing(
    child_partial: bool, zen_meta: dict, parents: Tuple[Type[DataClass_], ...]
):

    Conf = builds(
        dict, zen_partial=child_partial, zen_meta=zen_meta, builds_bases=parents
    )
    out = instantiate(Conf)
    if isinstance(out, partial):
        assert out.func is dict


@given(
    child_partial=st.none() | st.booleans(),
    zen_meta=st.sampled_from([{}, dict(meta=True)]),
    parents=parents_strat,
)
def test_partial_field_set_only_when_necessary(
    child_partial: bool, zen_meta: dict, parents: Tuple[Type[DataClass_], ...]
):
    Conf = builds(
        dict, zen_partial=child_partial, zen_meta=zen_meta, builds_bases=parents
    )
    # partial field should not be set unless necessary
    if child_partial is None and all(
        not hasattr(p, PARTIAL_FIELD_NAME) and not hasattr(p, ZEN_PARTIAL_FIELD_NAME)
        for p in parents
    ):
        assert not hasattr(Conf, PARTIAL_FIELD_NAME)
        assert not hasattr(Conf, ZEN_PARTIAL_FIELD_NAME)
    else:
        assert hasattr(Conf, PARTIAL_FIELD_NAME) or hasattr(
            Conf, ZEN_PARTIAL_FIELD_NAME
        )


@pytest.mark.parametrize(
    "parents, expected_partial",
    [
        ((), False),
        ((HydraNoPartial,), False),
        ((HydraPartialFalse,), False),
        ((HydraPartialTrue,), HYDRA_SUPPORTS_PARTIAL),
        ((ZenNoPartial,), False),
        ((ZenPartialFalse,), False),
        ((ZenPartialTrue,), True),
        ((ZenFalseHydraFalse,), False),
        ((ZenFalseHydraTrue,), HYDRA_SUPPORTS_PARTIAL),
        ((ZenTrueHydraFalse,), True),
        ((ZenTrueHydraTrue,), True),
        ((HydraNoPartial, HydraPartialFalse), False),
        ((HydraNoPartial, HydraPartialTrue), HYDRA_SUPPORTS_PARTIAL),
        ((HydraPartialFalse, HydraPartialTrue), False),
        ((HydraPartialTrue, HydraPartialFalse), HYDRA_SUPPORTS_PARTIAL),
        ((ZenNoPartial, HydraPartialTrue), HYDRA_SUPPORTS_PARTIAL),
        ((HydraPartialTrue, ZenPartialFalse), HYDRA_SUPPORTS_PARTIAL),
        ((HydraPartialTrue, ZenPartialTrue), True),
        ((HydraPartialFalse, ZenPartialTrue), not HYDRA_SUPPORTS_PARTIAL),
        ((ZenPartialFalse, HydraPartialTrue), False),
    ],
)
@pytest.mark.parametrize("meta", [{}, dict(meta=True)])
def test_partial_via_inheritance_explicit_cases(
    parents: Tuple[Type[DataClass_], ...],
    expected_partial: bool,
    meta,
):
    Conf = builds(int, zen_meta=meta, builds_bases=parents)
    assert is_partial_builds(Conf) is expected_partial


@pytest.mark.parametrize("Parent", [ZenPartialFalse, HydraPartialFalse])
def test_inherited_partial_false_fields(Parent):
    if not HYDRA_SUPPORTS_PARTIAL and Parent is HydraPartialFalse:
        pytest.mark.skip("Hydra version doesn't support instantiate API")

    Conf = builds(dict, builds_bases=(Parent,))  # should be OK
    assert instantiate(Conf) == instantiate(Parent)


def test_make_config_catches_multiple_inheritance_conflicting_with_zen_processing():
    @dataclass
    class A:
        _target_: str = "builtins.str"
        _partial_: bool = True

    B = builds(int, zen_partial=True, zen_meta=dict(a=1))

    with pytest.raises(ValueError):
        # Overwrites _target_=hydra_zen.funcs.zen_processing
        # leading to "corrupt" state
        _ = make_config(bases=(A, B))

    if HYDRA_SUPPORTS_PARTIAL:
        with pytest.raises(ValueError):
            # Specifies _partial_=True and _zen_partial=True
            _ = make_config(bases=(B, A))
