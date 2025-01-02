# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Generic, Sequence, Tuple, TypeVar

import pydantic
import pytest
from hydra.errors import InstantiationException
from pydantic import BaseModel, PositiveInt

from hydra_zen import BuildsFn, instantiate, zen
from hydra_zen.third_party.pydantic import pydantic_parser

T = TypeVar("T")


class MyBuilds(BuildsFn):
    def _sanitized_type(self, *args, **kwargs):
        # disable Hydra-level type checking
        return Any


builds = MyBuilds.builds
just = MyBuilds.just
instantiate = partial(instantiate, _target_wrapper_=pydantic_parser)


class Parent:
    y: str  # intentional: populates __annotations__

    def __init__(self, xoo: int) -> None:
        self.xoo = xoo

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.__dict__ == value.__dict__

    @classmethod
    def classmthd(cls, xoo: int) -> "Parent":
        return cls(xoo)

    @staticmethod
    def staticmthd(xoo: int) -> "Parent":
        return Parent(xoo)


class Child(Parent):
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


class HasGeneric(Generic[T]):
    def __init__(self, xoo: T):
        self.xoo = xoo

    def __repr__(self) -> str:
        return f"HasGeneric(xoo={self.xoo})"

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, type(self)):
            return False
        return self.xoo == value.xoo


@dataclass
class Param:
    target: Callable[..., Any]
    args: Sequence[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    msg: str = ""
    expected: Any = None


def p(
    target: Callable[..., Any], *args, msg: str = "", expected: Any = None, **kwargs
) -> Param:
    return Param(target, args, kwargs, expected=expected, msg=msg)


def no_annotate(xoo):
    return xoo


def func(xoo: int):
    return xoo + 2


@dataclass(eq=True, frozen=True)
class ADataClass:
    yee: str
    zaa: bool


class PydanticModel(BaseModel):
    xoo: int


@pytest.mark.parametrize(
    "obj",
    [
        p(Parent, xoo=10),
        p(Child, xoo=11, zoo=3.14),
        p(UsesNew, xoo=20),
        p(HasGeneric[int], xoo=21),
        p(HasGeneric, xoo="aa"),
        p(no_annotate, xoo=21),
        p(func, xoo=22),
        p(Parent.classmthd, xoo=23),
        p(Parent.staticmthd, xoo=24),
        p(partial(func, xoo=25)),
        p(ADataClass, yee="yee", zaa=True),
        p(len, [1, 2, 3]),  # func, no signature
        p(dict, a=1, b=2),  # class, no signature
        p(PydanticModel, xoo=1),
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
        p(Parent, xoo="aa", msg="xoo"),
        p(Child, xoo=10, zoo="aa", msg="zoo"),
        p(UsesNew, xoo="aa", msg="xoo"),
        pytest.param(
            p(no_annotate, xoo="aa"),
            marks=pytest.mark.xfail(reason="no annotation"),
        ),
        p(Parent.classmthd, xoo="a", msg="xoo"),
        p(Parent.staticmthd, xoo="bv", msg="xoo"),
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


@pytest.mark.parametrize(
    "obj",
    [
        pytest.param(
            p(g, x=builds(A2, x=[1, 2, 3]), expected=A2(x=(1, 2, 3))),
            marks=pytest.mark.skipif(
                pydantic.VERSION < "2.0", reason="pydantic>=2.0 required"
            ),
        ),
        p(h, x=".", expected=Path(".")),
    ],
)
def test_conversion_support(obj: Param):
    cfg = builds(obj.target, *obj.args, **obj.kwargs)
    out = instantiate(cfg)
    assert out == obj.expected


def int_(yoo: PositiveInt):
    return yoo


def pos(xoo: PositiveInt):
    return xoo


def test_with_zen():
    assert zen(pos, instantiation_wrapper=pydantic_parser)(dict(xoo=3)) == 3

    # test parsing on zen-wrapped function
    with pytest.raises(
        (InstantiationException, pydantic.ValidationError),
        match="xoo",
    ):
        zen(pos, instantiation_wrapper=pydantic_parser)(dict(xoo=-3))

    good_cfg = builds(int_, yoo=8)

    assert zen(pos, instantiation_wrapper=pydantic_parser)(dict(xoo=good_cfg)) == 8

    # test parsing on config passed to zen-wrapped function
    bad_cfg = builds(int_, yoo=-9)

    with pytest.raises(
        (InstantiationException, pydantic.ValidationError),
        match="yoo",
    ):
        zen(pos, instantiation_wrapper=pydantic_parser)(dict(xoo=bad_cfg))


async def async_func(xoo: int):
    return xoo


async def test_async_support():
    out = await zen(async_func, instantiation_wrapper=pydantic_parser)(dict(xoo=1))
    assert out == 1

    with pytest.raises(
        (InstantiationException, pydantic.ValidationError),
        match="xoo",
    ):
        await zen(async_func, instantiation_wrapper=pydantic_parser)(dict(xoo="aaa"))
