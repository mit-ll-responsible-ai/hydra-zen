import sys
from dataclasses import dataclass
from typing import Dict, List

import hypothesis.strategies as st
import omegaconf
import pytest
from hypothesis import given, note
from omegaconf import OmegaConf

from hydra_utils import mutable_value
from hydra_utils.structured_configs._utils import interpolated, safe_name

from . import valid_hydra_literals

current_module: str = sys.modules[__name__].__name__


def pass_through(*args):
    return args


def pass_through_kwargs(**kwargs):
    return kwargs


omegaconf.OmegaConf.register_new_resolver("_test_pass_through", pass_through)


@given(st.lists(valid_hydra_literals))
def test_interpolate_roundtrip(literals):
    interpolated_string = interpolated("_test_pass_through", *literals)

    note(interpolated_string)

    interpolated_literals = OmegaConf.create({"x": interpolated_string}).x

    assert len(literals) == len(interpolated_literals)

    for lit, interp in zip(literals, interpolated_literals):
        assert lit == interp


class C:
    def __repr__(self):
        return "C as a repr"

    def f(self):
        return


def f():
    pass


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (1, "1"),
        (dict, "dict"),
        (C, "C"),
        (C.f, "C.f"),
        (C(), "C as a repr"),
        ("moo", "'moo'"),
        (None, "None"),
        (f, "f"),
    ],
)
def test_safename_known(obj, expected_name):
    assert safe_name(obj) == expected_name


@given(
    st.from_type(type)
)  # this draws any type that has a strategy registered with hypothesis!
def test_fuzz_safename(obj):
    safe_name(obj)


def test_mutable_values():
    @dataclass
    class A:
        a_list: List[int] = mutable_value([1, 2, 3])
        a_dict: Dict[str, int] = mutable_value(dict(a=1))

    a = A()
    assert a.a_dict == {"a": 1}
    assert a.a_list == [1, 2, 3]


def test_documented_instantiate_example():
    from hydra_utils import builds, instantiate

    assert instantiate(builds(dict, a=1, b=2), c=3) == dict(a=1, b=2, c=3)
    assert instantiate(builds(list), (1, 2, 3)) == [1, 2, 3]
