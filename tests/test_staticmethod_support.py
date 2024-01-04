# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest

from hydra_zen import builds, instantiate, just


class A1:
    @staticmethod
    def foo(x: int):
        return "A.foo" * x


class B1(A1):
    pass


class C1(A1):
    @staticmethod
    def foo(x: int):
        return "C.foo" * x


class A2:
    class B2:
        @staticmethod
        def f():
            return "nested"


def test_builds_static_methods():
    assert instantiate(builds(A1.foo, 2)) == "A.foo" * 2
    assert instantiate(builds(B1.foo, 3)) == "A.foo" * 3
    assert instantiate(builds(C1.foo, 4)) == "C.foo" * 4


@pytest.mark.parametrize("obj", [C1.foo, A2.B2.f])
def test_just_static_method(obj):
    assert instantiate(just(obj)) is obj


def some_func():
    return 22


@pytest.mark.parametrize("qualname", ["a", "a.", "1a.a"])
def test_that_we_dont_look_at_qualname_unless_it_looks_like_a_path(qualname):
    some_func.__qualname__ = qualname
    assert instantiate(builds(some_func)) == 22
