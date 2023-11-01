# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import pytest

from hydra_zen import BuildsFn, builds, instantiate, note_static_method


class MyBuilds(BuildsFn):
    _registered_static_methods = set()


class A:
    @staticmethod
    def not_noted():
        ...

    @staticmethod
    def foo(x: int):
        return "A.foo" * x


class B(A):
    pass


class C(A):
    @staticmethod
    def foo(x: int):
        return "C.foo" * x


note_static_method(A.foo)
note_static_method(C.foo)


def test_noting_is_necessary():
    with pytest.raises(Exception, match="Error locating target"):
        instantiate(builds(A.not_noted))


def test_builds_static_methods():
    assert instantiate(builds(A.foo, 2)) == "A.foo" * 2
    assert instantiate(builds(B.foo, 3)) == "A.foo" * 3
    assert instantiate(builds(C.foo, 4)) == "C.foo" * 4


class BB:
    @staticmethod
    def f():
        return "moo"


def test_local_registry_is_isolated():
    BuildsFn.note_static_method(A.foo)
    num_global_registered = len(BuildsFn._registered_static_methods)
    assert num_global_registered

    # `MyBuilds` should not know about `A.foo`
    with pytest.raises(Exception, match="Error locating target"):
        instantiate(MyBuilds.builds(A.foo))

    with pytest.raises(Exception, match="Error locating target"):
        instantiate(MyBuilds.builds(BB.f))

    assert instantiate(MyBuilds.builds(MyBuilds.note_static_method(BB.f))) == "moo"
    assert num_global_registered == len(BuildsFn._registered_static_methods)


def test_validation():
    with pytest.raises(TypeError):
        note_static_method(BB)
