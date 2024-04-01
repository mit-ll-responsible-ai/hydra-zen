# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Sequence, Tuple

import pydantic
import pytest
from hydra.errors import InstantiationException

from hydra_zen import BuildsFn, instantiate
from hydra_zen.third_party.pydantic import with_pydantic_parsing


class MyBuilds(BuildsFn):
    def _sanitized_type(self, *args, **kwargs):
        # disable Hydra-level type checking
        return Any


builds = MyBuilds.builds
just = MyBuilds.just
instantiate = partial(instantiate, _target_wrapper_=with_pydantic_parsing)


class A:
    y: str  # intentional: populates __annotations__

    def __init__(self, xoo: int) -> None:
        self.xoo = xoo

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.__dict__ == value.__dict__

    @classmethod
    def classmthd(cls, xoo: int) -> "A":
        return cls(xoo)

    @staticmethod
    def staticmthd(xoo: int) -> "A":
        return A(xoo)


class B(A):
    def __init__(self, xoo: int, zoo: float) -> None:
        super().__init__(xoo)
        self.zoo = zoo


class UsesNew:
    def __new__(cls, xoo: int):
        self = object.__new__(cls)
        self.xoo = xoo  # type: ignore
        return self

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.__dict__ == value.__dict__


@dataclass
class Param:
    target: Callable[..., Any]
    args: Sequence[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    msg: str = ""


def p(target: Callable[..., Any], *args, msg: str = "", **kwargs) -> Param:
    return Param(target, args, kwargs)


def no_annotate(xoo):
    return xoo


def func(xoo: int):
    return xoo + 2


@dataclass(eq=True, frozen=True)
class ADataClass:
    yee: str
    zaa: bool


@pytest.mark.parametrize(
    "obj",
    [
        p(A, xoo=10),
        p(B, xoo=11, zoo=3.14),
        p(UsesNew, xoo=20),
        p(no_annotate, xoo=21),
        p(func, xoo=22),
        p(A.classmthd, xoo=23),
        p(A.staticmthd, xoo=24),
        p(partial(func, xoo=25)),
        p(ADataClass, yee="yee", zaa=True),
        p(len, [1, 2, 3]),  # func, no signature
        p(dict, a=1, b=2),  # class, no signature
    ],
)
@pytest.mark.parametrize("use_meta_feature", [True, False])
def test_roundtrip(obj: Param, use_meta_feature: bool):
    kw = obj.kwargs.copy()
    if use_meta_feature:
        kw["zen_meta"] = {"_za": 1}
    inst_out = instantiate(builds(obj.target, *obj.args, **kw))
    out = obj.target(*obj.args, **obj.kwargs)

    if inspect.isclass(obj.target):
        assert isinstance(inst_out, obj.target)
    assert out == inst_out


@pytest.mark.parametrize(
    "obj",
    [
        p(A, xoo="aa", msg="xoo"),
        p(B, xoo=10, zoo="aa", msg="zoo"),
        p(UsesNew, xoo="aa", msg="xoo"),
        pytest.param(
            p(no_annotate, xoo="aa"),
            marks=pytest.mark.xfail(reason="no annotation"),
        ),
        p(A.classmthd, xoo="a", msg="xoo"),
        p(A.staticmthd, xoo="bv", msg="xoo"),
        p(func, xoo=[1], msg="xoo"),
        p(ADataClass, yee="yee", zaa=(1,), msg="zaa"),
    ],
)
@pytest.mark.parametrize("use_meta_feature", [True, False])
def test_type_checking(obj: Param, use_meta_feature: bool):
    kw = obj.kwargs.copy()

    if use_meta_feature:
        kw["zen_meta"] = {"a": 1}
    cfg = builds(obj.target, *obj.args, **kw)

    with pytest.raises(
        (InstantiationException, pydantic.ValidationError),
        match=obj.msg,
    ):
        instantiate(cfg)


@dataclass
class A2:
    x: Tuple[int, int, int]


def g(x: A2):
    return x


def h(x: Path):
    return x


def test_conversion_support():
    assert instantiate(builds(g, x=builds(A2, x=[1, 2, 3]))) == A2(x=(1, 2, 3))
    assert instantiate(builds(h, x=".")) == Path(".")
