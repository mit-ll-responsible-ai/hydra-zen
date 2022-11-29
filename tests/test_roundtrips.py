# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import datetime
import math
import operator
import os
import pickle
import random
import re
import statistics
import string
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf

from hydra_zen import builds, get_target, hydrated_dataclass, instantiate, just, to_yaml
from hydra_zen.structured_configs._type_guards import is_builds
from tests import is_same, valid_hydra_literals

arbitrary_kwargs = st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=1),
    values=valid_hydra_literals,
)


def pass_through_kwargs(**kwargs):
    return kwargs


def pass_through_args(*args):
    return args


@given(kwargs=arbitrary_kwargs, full_sig=st.booleans())
def test_builds_roundtrip(kwargs, full_sig: bool):
    assert kwargs == instantiate(
        builds(pass_through_kwargs, **kwargs, populate_full_signature=full_sig)
    )


@given(
    partial_kwargs=arbitrary_kwargs,
    call_kwargs=arbitrary_kwargs,
    full_sig=st.booleans(),
)
def test_builds_kwargs_roundtrip_with_partial(
    partial_kwargs: Dict[str, Any],
    call_kwargs: Dict[str, Any],
    full_sig: bool,
):
    partial_struct = instantiate(
        builds(
            pass_through_kwargs,
            zen_partial=True,
            populate_full_signature=full_sig,
            **partial_kwargs,
        )
    )
    expected_kwargs = partial_kwargs.copy()
    expected_kwargs.update(call_kwargs)
    assert expected_kwargs == partial_struct(**call_kwargs)  # resolve partial


@given(
    partial_args=arbitrary_kwargs.map(lambda x: list(x.values())),
    call_args=arbitrary_kwargs.map(lambda x: list(x.values())),
    full_sig=st.booleans(),
)
def test_builds_args_roundtrip_with_partial(
    partial_args: List[Any],
    call_args: List[Any],
    full_sig: bool,
):
    partial_struct = instantiate(
        builds(
            pass_through_args,
            zen_partial=True,
            populate_full_signature=full_sig,
            *partial_args,
        ),
    )

    expected_args = partial_args.copy()
    expected_args.extend(call_args)
    assert tuple(expected_args) == partial_struct(*call_args)  # resolve partial


def f(x, y=dict(a=2)):
    return x, y


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False, None])
@pytest.mark.parametrize("named_arg", [True, False])
def test_builds_roundtrips_with_mutable_values(
    full_sig: bool, partial: Optional[bool], named_arg: bool
):
    # tests mutable user-specified value and default value
    if named_arg:
        result = instantiate(
            builds(f, x=[1], populate_full_signature=full_sig, zen_partial=partial)
        )
    else:
        result = instantiate(
            builds(f, [1], populate_full_signature=full_sig, zen_partial=partial)
        )
    if partial:
        result = result()
    assert result == ([1], {"a": 2})


class LocalClass:
    @classmethod
    def a_class_method(cls):
        return


def local_function():
    pass


a_bunch_of_objects = [
    local_function,
    LocalClass,
    LocalClass.a_class_method,
    int,
    str,
    list,
    set,
    complex,
    isinstance,
    all,
    Exception,
    random.random,
    random.uniform,
    random.choice,
    random.choices,
    re.compile,
    re.match,
    datetime.time,
    datetime.timezone,
    math.sin,
    operator.add,
    statistics.mean,
    os.getcwd,
    Counter,
    deque,
    defaultdict,
]


def with_is(x, y):
    return x is y


def with_eq(x, y):
    return x == y


@pytest.mark.parametrize("obj", a_bunch_of_objects)
def test_just_roundtrip(obj):
    # local classmethods have weird identity behaviors
    cfg = just(obj)
    assert is_same(instantiate(cfg), obj)
    assert is_same(instantiate(OmegaConf.create(to_yaml(cfg))), obj)


@pytest.mark.parametrize("x", a_bunch_of_objects)
@pytest.mark.parametrize(
    "fn",
    [
        builds,
        just,
        lambda x: builds(x, zen_partial=True),
        lambda x: builds(x, zen_meta=dict(_some_obscure_name=1)),
        lambda x: builds(x, zen_partial=True, zen_meta=dict(_some_obscure_name=1)),
    ],
)
def test_get_target_roundtrip(x, fn):
    conf = fn(x)
    assert is_same(x, get_target(conf))

    loaded = OmegaConf.create(to_yaml(conf))
    assert is_builds(loaded)
    assert is_same(x, get_target(loaded))


@dataclass
class A:
    _target_: Any = int


def test_get_target_with_non_string_target():
    assert get_target(A) is int


def test_recursive_just():
    x = {"a": [3 - 4j, 1 + 2j]}
    assert instantiate(just(x)) == x


def fn_target(x: int, y: int):
    return x + y


@hydrated_dataclass(fn_target)
class HydratedExample:
    x: int
    y: int = 3


def test_hydrated_dataclass_pickles():
    assert pickle.loads(pickle.dumps(HydratedExample)) is HydratedExample
    assert pickle.loads(pickle.dumps(HydratedExample(x=2))) == HydratedExample(x=2)
