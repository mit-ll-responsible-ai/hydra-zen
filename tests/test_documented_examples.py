# Copyright (c) 2022 achusetts Institute of Technology
# SPDX-License-Identifier: MIT
import pytest

from hydra_zen import builds, hydrated_dataclass, instantiate, just, to_yaml


def test_hydrated_simple_example():
    @hydrated_dataclass(target=dict)
    class DictConf:
        x: int = 2
        y: str = "hello"

    assert instantiate(DictConf(x=10)) == dict(x=10, y="hello")


def power(x: float, exponent: float) -> float:
    return x**exponent


def test_hydrated_with_partial_exampled():
    @hydrated_dataclass(target=power, zen_partial=True)
    class PowerConf:
        exponent: float = 2.0

    partiald_power = instantiate(PowerConf)
    assert partiald_power(10.0) == 100.0


def test_documented_builds_simple_roundtrip_example():
    assert {"a": 1, "b": "x"} == instantiate(builds(dict, a=1, b="x"))


def test_documented_builds_pos_args_roundtrip_example():
    assert [1, 2, 3] == instantiate(builds(list, (1, 2, 3)))


def f():
    pass


class C:
    @classmethod
    def f(cls, x):
        return x


def func(a_class, a_list=[1, 2], a_func=f, a_builtin_func=len):
    return a_class, a_list, a_func, a_builtin_func


@pytest.mark.parametrize("as_pos_arg", [True, False])
def test_auto_normalization_of_default_values(as_pos_arg):
    if as_pos_arg:
        conf = builds(func, C, populate_full_signature=True)
    else:
        conf = builds(func, a_class=C, populate_full_signature=True)
    to_yaml(conf)  # raises if not serializable
    a_class, a_list, a_func, a_builtin_func = instantiate(conf)
    assert a_class is C
    assert a_list == [1, 2]
    assert a_func is f
    assert a_builtin_func is len


def test_nested_configs():
    d = instantiate(builds(dict, x=builds(dict, w=1), y=just(f), z=f))
    assert d["x"] == dict(w=1)
    assert d["y"] is f
    assert d["z"] is f


def test_nested_configs_pos_args():
    x, y, z = instantiate(builds(list, (builds(dict, w=1), just(f), f)))
    assert x == dict(w=1)
    assert y is f
    assert z is f


def test_builds_classmethod():
    assert instantiate(builds(C.f, x=10)) == 10
