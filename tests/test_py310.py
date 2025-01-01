# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import sys

assert sys.version_info > (3, 9)

from dataclasses import KW_ONLY, dataclass

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

from hydra_zen import builds, instantiate, just, make_config
from hydra_zen.structured_configs._type_guards import safe_getattr
from hydra_zen.typing import DataclassOptions as Dc
from tests.custom_strategies import valid_builds_args

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


def func(x): ...


@pytest.mark.filterwarnings("ignore:A structured config was supplied for")
@given(kw=valid_builds_args(), as_inst=..., x=st.sampled_from(["", 1, [1, 2]]))
def test_equiv_getattr(kw, as_inst: bool, x):
    obj = builds(func, x=x, **kw)

    if "zen_dataclass" in kw and kw["zen_dataclass"] is not None:
        dc_opts = kw["zen_dataclass"]
        if dc_opts.get("slots", False) and not dc_opts.get("init", True):
            # I think @dataclass(slots=True, init=False) has buggy
            # behavior!!
            assume(False)

        kw["zen_dataclass"]["slots"] = False
        kw["zen_dataclass"]["weakref_slot"] = False

    no_slot_obj = builds(func, x=x, **kw)

    if as_inst:
        obj = obj()
        no_slot_obj = no_slot_obj()

    try:
        getattr(no_slot_obj, "x")
    except AttributeError:
        with pytest.raises(AttributeError):
            safe_getattr(obj, "x")
        return

    actual = safe_getattr(obj, "x")

    assert x == actual


def test_safe_getattr_no_default():
    conf = make_config("x", zen_dataclass={"slots": True})

    with pytest.raises(AttributeError):
        safe_getattr(conf, "x")

    assert safe_getattr(conf, "x", 1) == 1
