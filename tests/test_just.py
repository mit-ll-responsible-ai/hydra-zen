# Copyright (c) 2022 achusetts Institute of Technology
# SPDX-License-Identifier: MIT
import functools
from dataclasses import dataclass

import pytest

from hydra_zen import builds, instantiate, just, make_config, to_yaml
from tests import is_same


class A:
    @classmethod
    def class_method(cls):
        pass


@dataclass
class ADataclass:
    x: int


def f(x: int):
    pass


@functools.lru_cache(maxsize=None)
def func_with_cache(x: int):
    pass


list_of_objects = [
    bytearray([1, 2, 3]),
    1 + 2j,
    A,
    f,
    func_with_cache,
    A.class_method,
    functools.partial(f, x=1),
    ADataclass,
    ADataclass(x=2),
]


# this function is tested more rigorously via test_value_conversion
@pytest.mark.parametrize("obj", list_of_objects)
def test_just_roundtrip(obj):
    out = instantiate(just(obj))

    if callable(out):
        assert is_same(out, obj)
    else:
        assert out == obj


@pytest.mark.parametrize("obj", list_of_objects)
def test_just_idempotence_via_instantiate(obj):
    expected = instantiate(just(obj))
    actual = instantiate(just(just(obj)))
    if callable(expected):
        assert is_same(actual, expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("obj", list_of_objects)
def test_just_idempotence_via_yaml(obj):
    expected = to_yaml(just(obj))
    actual = to_yaml(just(just(obj)))
    assert actual == expected


def test_just_of_targeted_config_is_identity():
    cfg = builds(dict, x=1)
    assert just(cfg) is cfg


def test_just_no_dataclass_autoconfig():
    Cfg = make_config()
    instt = Cfg()
    assert just(Cfg, zen_convert={"dataclass": False}) is Cfg
    assert just(instt, zen_convert={"dataclass": False}) is instt


@pytest.mark.parametrize(
    "kwargs", [{"hydra_convert": "mall"}, {"hydra_recursive": make_config()}]
)
def test_validate_hydra_options(kwargs):
    with pytest.raises((ValueError, TypeError)):
        just(ADataclass(x=1), **kwargs)
