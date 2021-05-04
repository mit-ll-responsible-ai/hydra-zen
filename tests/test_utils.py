# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import collections.abc as abc
import enum
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

import hypothesis.strategies as st
import omegaconf
import pytest
from hypothesis import given, note
from omegaconf import OmegaConf, ValidationError
from omegaconf.errors import (
    ConfigIndexError,
    ConfigTypeError,
    ConfigValueError,
    KeyValidationError,
)
from typing_extensions import Final, Literal, Protocol

from hydra_zen import builds, instantiate, mutable_value
from hydra_zen.structured_configs._utils import interpolated, safe_name, sanitized_type

from . import valid_hydra_literals

T = TypeVar("T")

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


OmegaConf.register_new_resolver("len", len)


def test_interpolation_with_string_literal():
    cc = instantiate(builds(dict, total=interpolated(len, "9")))
    assert cc["total"] == 1


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
    from hydra_zen import builds, instantiate

    assert instantiate(builds(dict, a=1, b=2), c=3) == dict(a=1, b=2, c=3)
    assert instantiate(builds(list), (1, 2, 3)) == [1, 2, 3]


class Color(enum.Enum):
    pass


class SomeProtocol(Protocol):
    pass


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),  # supported primitives
        (float, float),
        (str, str),
        (bool, bool),
        (Color, Color),
        (C, Any),  # unsupported primitives
        (set, Any),
        (list, Any),
        (tuple, Any),
        (dict, Any),
        (callable, Any),
        (frozenset, Any),
        (T, Any),
        (Literal[1, 2], Any),  # unsupported generics
        (Type[int], Any),
        (SomeProtocol, Any),
        (Set, Any),
        (Set[int], Any),
        (Final[int], Any),
        (Callable, Any),
        (Callable[[int], int], Any),
        (abc.Callable, Any),
        (abc.Mapping, Any),
        (Union[str, int], Any),
        (Optional[frozenset], Any),
        (Union[type(None), frozenset], Any),
        (Union[type(None), int], Optional[int]),  # supported Optional
        (Optional[Color], Optional[Color]),
        (Optional[List[Color]], Optional[List[Color]]),
        (Optional[List[List[int]]], Optional[List[Any]]),
        (List[int], List[int]),  # supported containers
        (List[frozenset], List[Any]),
        (List[List[int]], List[Any]),
        (List[T], List[Any]),
        (Dict[str, float], Dict[str, float]),
        (Dict[C, int], Dict[Any, int]),
        (Dict[str, C], Dict[str, Any]),
        (Dict[C, C], Dict[Any, Any]),
        (Dict[str, List[int]], Dict[str, Any]),
        (Tuple[str, str, str], Tuple[str, str, str]),
        (Tuple[List[int]], Tuple[Any]),
    ],
)
def test_sanitized_type_expected_behavior(in_type, expected_type):

    assert sanitized_type(in_type) == expected_type, in_type

    if in_type != expected_type:
        # In cases where we change the type, it should be because omegaconf
        # doesn't support that annotation.
        # This check will help catch cases where omegaconf/hydra has added support for
        # new annotations like Literal
        @dataclass
        class Bad:
            x: in_type

        with pytest.raises(
            (
                ValidationError,
                AssertionError,
                ConfigTypeError,
                KeyValidationError,
                ConfigIndexError,
                ConfigValueError,
            )
        ):
            OmegaConf.create(Bad)

    @dataclass
    class Tmp:
        x: expected_type

    OmegaConf.structured(Tmp)  # shouldn't raise on `expected_type`


def test_tuple_annotation_normalization():
    assert sanitized_type(Tuple[int, str, int]) is Tuple[Any, Any, Any]


def f_list(x: List):
    return x


def f_dict(x: Dict):
    return x


def f_tuple(x: Tuple):
    return x


@pytest.mark.parametrize(
    "func, value", [(f_list, [1]), (f_dict, dict(a=1)), (f_tuple, (1,))]
)
def test_bare_generics(func, value):
    # python is super flaky about how it represents bare generics.. so
    # we have to write a separate test for them here where we don't
    # check the output of sanitize_type
    assert instantiate(builds(func, populate_full_signature=True, x=value)) == value
