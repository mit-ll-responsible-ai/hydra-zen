# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Literal, Tuple, Type

from omegaconf import MISSING, DictConfig, ListConfig

from hydra_zen import (
    ZenField,
    builds,
    get_target,
    instantiate,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
)
from hydra_zen.typing import Builds


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


# test type behaviors
def f1():
    # test builds(..., zen_partial=True)
    # there is something really weird where annotating : PartialBuilds[Type[A]] breaks this..
    a: Literal["Type[PartialBuilds[Type[A]]]"] = reveal_type(
        builds(A, zen_partial=True)
    )
    conf_a_partial = builds(A, zen_partial=True)
    b: Literal["Partial[A]"] = reveal_type(instantiate(conf_a_partial))
    c: Literal["A"] = reveal_type(instantiate(conf_a_partial)())


f_sig = Callable[[int], int]


def f2():
    a: Literal["Type[PartialBuilds[(x: int) -> int]]"] = reveal_type(
        builds(f, zen_partial=True)
    )

    conf_f_partial = builds(f, zen_partial=True)

    b: Literal["PartialBuilds[(x: int) -> int]"] = reveal_type(conf_f_partial())

    conf_f_partial_instance = conf_f_partial()
    c: Literal["Partial[int]"] = reveal_type(instantiate(conf_f_partial))
    d: Literal["Partial[int]"] = reveal_type(instantiate(conf_f_partial_instance))
    e: Literal["int"] = reveal_type(instantiate(conf_f_partial)())


def f3():
    # test builds(..., zen_partial=False)
    a: Literal["Type[Builds[Type[A]]]"] = reveal_type(builds(A, zen_partial=False))
    conf_A_1 = builds(A, zen_partial=False)
    b: Literal["A"] = reveal_type(instantiate(conf_A_1))

    c: Literal["Type[Builds[(x: int) -> int]]"] = reveal_type(
        builds(f, zen_partial=False)
    )
    conf_f_1: Type[Builds[f_sig]] = builds(f, zen_partial=False)
    d: Literal["int"] = reveal_type(instantiate(conf_f_1))


def f4():
    # test builds(...)
    a: Literal["Type[Builds[Type[A]]]"] = reveal_type(builds(A))
    conf_A_2 = builds(A)
    b: Literal["A"] = reveal_type(instantiate(conf_A_2))

    c: Literal["Builds[Type[A]]"] = reveal_type(conf_A_2())
    conf_a_instance = conf_A_2()
    d: Literal["A"] = reveal_type(instantiate(conf_a_instance))

    e: Literal["Type[Builds[(x: int) -> int]]"] = reveal_type(builds(f))
    conf_f_2 = builds(f)
    ee: Literal["int"] = reveal_type(instantiate(conf_f_2))


def f5():
    # test just(...)
    a: Literal["Type[Just[(x: int) -> int]]"] = reveal_type(just(f))
    b: Literal["Type[Just[Type[A]]]"] = reveal_type(just(A))
    c: Literal["(x: int) -> int"] = reveal_type(instantiate(just(f)))
    d: Literal["Type[A]"] = reveal_type(instantiate(just(A)))
    e: Literal["Type[A]"] = reveal_type(instantiate(just(A)()))  # instance of Just


@dataclass
class SomeDataClass:
    pass


def f6():
    some_dataclass = SomeDataClass()

    out1 = instantiate(SomeDataClass)
    out2 = instantiate(some_dataclass)


def f7():
    # get_target(Type[Builds[T]]) -> T
    a1: Literal["Type[str]"] = reveal_type(get_target(builds(str)))
    a2: Literal["Type[str]"] = reveal_type(get_target(builds(str, zen_partial=False)))
    a3: Literal["Type[str]"] = reveal_type(get_target(builds(str, zen_partial=True)))
    a4: Literal["Type[str]"] = reveal_type(get_target(just(str)))

    # get_target(Builds[Callable[...]]) -> Callable[...]
    b1: Literal["(x: int) -> int"] = reveal_type(get_target(builds(f)))
    b2: Literal["(x: int) -> int"] = reveal_type(
        get_target(builds(f, zen_partial=False))
    )
    b3: Literal["(x: int) -> int"] = reveal_type(
        get_target(builds(f, zen_partial=True))
    )
    b4: Literal["(x: int) -> int"] = reveal_type(get_target(just(f)))

    c1: Literal["Type[str]"] = reveal_type(get_target(builds(str)()))
    c2: Literal["Type[str]"] = reveal_type(get_target(builds(str, zen_partial=False)()))
    c3: Literal["Type[str]"] = reveal_type(get_target(builds(str, zen_partial=True)()))
    c4: Literal["Type[str]"] = reveal_type(get_target(just(str)()))


def f8():
    @dataclass
    class A:
        x: List[int] = mutable_value([1, 2])


def zen_wrappers():
    def f(obj):
        return obj

    J = just(f)
    B = builds(f, zen_partial=True)
    PB = builds(f, zen_partial=True)
    a1: Literal["Type[Builds[Type[str]]]"] = reveal_type(builds(str, zen_wrappers=f))
    a2: Literal["Type[Builds[Type[str]]]"] = reveal_type(builds(str, zen_wrappers=J))
    a3: Literal["Type[Builds[Type[str]]]"] = reveal_type(builds(str, zen_wrappers=B))
    a4: Literal["Type[Builds[Type[str]]]"] = reveal_type(builds(str, zen_wrappers=PB))
    a5: Literal["Type[Builds[Type[str]]]"] = reveal_type(
        builds(str, zen_wrappers=(None,))
    )

    a6: Literal["Type[Builds[Type[str]]]"] = reveal_type(
        builds(str, zen_wrappers=(f, J, B, PB, None))
    )

    b1: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=f)
    )
    b2: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=J)
    )
    b3: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=B)
    )
    b4: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=PB)
    )
    b5: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(None,))
    )
    b6: Literal["Type[PartialBuilds[Type[str]]]"] = reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(f, J, B, PB, None))
    )


def custom_builds_fn():
    _builds = make_custom_builds_fn()

    a1: Literal["Type[Builds[Type[int]]]"] = reveal_type(_builds(int))
    a2: Literal["Type[PartialBuilds[Type[int]]]"] = reveal_type(
        _builds(int, zen_partial=True)
    )


def supported_primitives():
    class M:
        pass

    def f(*args):
        pass

    @dataclass
    class ADataclass:
        x: int = 1

    class AnEnum(Enum):
        a = 1
        b = 2

    olist = ListConfig([1, 2, 3])
    odict = DictConfig({"1": 1})

    a1: Literal["Type[DataClass]"] = reveal_type(
        make_config(
            a=(
                1,
                "hi",
                2.0,
                1j,
                set(),
                M,
                ADataclass,
                builds(dict),
                Path.cwd(),
                olist,
                odict,
                AnEnum.a,
            ),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
            f=ADataclass(),
            g=builds(int)(),  # dataclass instance
            h=builds(int, zen_partial=True)(),  # dataclass instance
        )
    )
    a2: Literal["Type[DataClass]"] = reveal_type(
        make_config(
            ZenField(name="a", default={M}),
            ZenField(name="b", default={1: M}),
            ZenField(name="c", default=[2.0 + 1j]),
            d=ZenField(default=(1, "hi", 2.0, 1j, set(), M, Path.cwd())),
            e=ZenField(default=f),
        )
    )

    a3: Literal["Type[Builds[Type[dict[Unknown, Unknown]]]]"] = reveal_type(
        builds(
            dict,
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(dict), Path.cwd()),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
        )
    )

    a4: Literal["Type[PartialBuilds[Type[dict[Unknown, Unknown]]]]"] = reveal_type(
        builds(
            dict,
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(dict), Path.cwd()),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
            zen_partial=True,
        )
    )

    # check lists
    a5 = make_config(a=[], b=[1], c=[[1]], d=[[[M]]])

    # check dicts
    a6 = make_config(
        a={}, b={1: 1}, c=[{1: 1}], d={1: {"a": "a"}}, e={"a": 1j}, f={"a": [1j]}
    )

    a7 = builds(
        f,
        None,
        MISSING,
        1,
        "hi",
        2.0,
        1j,
        M,
        ADataclass,
        builds(dict),
        Path.cwd(),
        set(),
        frozenset(),
        {1, 1j, Path.cwd()},
        deque(),
        Counter(),
        [deque(), Counter(), 1j],
        (deque(), Counter(), 1j),
        range(1, 10, 2),
        odict,
        olist,
    )

    a_list = [1, 2, [1, 2]]
    a_dict = {"a": [1, 2, [1, 2]]}
    a_set = {1, 2.0, (1, 2)}

    # make sure we don't hit this issue again
    # https://github.com/microsoft/pyright/issues/2659
    a8 = make_config(x=a_list, y=a_dict, z=a_set)

    # The following should be marked as "bad by type-checkers
    # make_config(a=M())
    # make_config(a=(1, M()))
    # builds(dict, a=M())

    # The following *should* be invalid, but we are limited
    # by mutable invariants being generic
    # make_config(a={1j: 1})
    # make_config(a={M: 1})
    # make_config(a={ADataclass: 1})
    # make_config(a={ADataclass(): 1})
