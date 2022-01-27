# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

# These tests help to ensure that our typed interfaces have the desired behvarior, when
# being processed by static type-checkers. Specifically we test using pyright.
#
# We perform contrapositive testing using lines with the pattern:
#
#  builds(dict, a=M())  # type: ignore
#
# We are testing that the type-checker raises an error on that line of code.
# We achieve this by configuring pyright to raise whenever `# type: ignore`
# is included unnecessarily. Thus we are ensuring that the type-checker does
# indeed need to ignore an error on that line.

from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Callable, List, Tuple, Type, TypeVar

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
from hydra_zen.typing import Builds, Partial
from hydra_zen.typing._implementations import HydraPartialBuilds

T = TypeVar("T")


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


# test type behaviors
def f1():
    reveal_type(
        builds(A, zen_partial=True), expected_text="Type[PartialBuilds[Type[A]]]"
    )
    conf_a_partial = builds(A, zen_partial=True)
    reveal_type(instantiate(conf_a_partial), expected_text="Partial[A]")
    reveal_type(instantiate(conf_a_partial)(), expected_text="A")


f_sig = Callable[[int], int]


def f2():
    reveal_type(
        builds(f, zen_partial=True),
        expected_text="Type[PartialBuilds[(x: int) -> int]]",
    )

    conf_f_partial = builds(f, zen_partial=True)

    reveal_type(conf_f_partial(), expected_text="PartialBuilds[(x: int) -> int]")

    conf_f_partial_instance = conf_f_partial()
    reveal_type(instantiate(conf_f_partial), expected_text="Partial[int]")
    reveal_type(instantiate(conf_f_partial_instance), expected_text="Partial[int]")
    reveal_type(instantiate(conf_f_partial)(), expected_text="int")


def f3():
    # test builds(..., zen_partial=False)
    reveal_type(builds(A, zen_partial=False), expected_text="Type[Builds[Type[A]]]")
    conf_A_1 = builds(A, zen_partial=False)
    reveal_type(instantiate(conf_A_1), expected_text="A")

    reveal_type(
        builds(f, zen_partial=False, expected_text="Type[Builds[(x: int) -> int]]")
    )
    conf_f_1: Type[Builds[f_sig]] = builds(f, zen_partial=False)
    reveal_type(instantiate(conf_f_1), expected_text="int")


def f4():
    # test builds(...)
    reveal_type(builds(A), expected_text="Type[Builds[Type[A]]]")
    conf_A_2 = builds(A)
    reveal_type(instantiate(conf_A_2), expected_text="A")

    reveal_type(conf_A_2(), expected_text="Builds[Type[A]]")
    conf_a_instance = conf_A_2()
    reveal_type(instantiate(conf_a_instance), expected_text="A")

    reveal_type(builds(f), expected_text="Type[Builds[(x: int) -> int]]")
    conf_f_2 = builds(f)
    reveal_type(instantiate(conf_f_2), expected_text="int")


def f5():
    # test just(...)
    reveal_type(just(f), expected_text="Type[Just[(x: int) -> int]]")
    reveal_type(just(A), expected_text="Type[Just[Type[A]]]")
    reveal_type(instantiate(just(f)), expected_text="(x: int) -> int")
    reveal_type(instantiate(just(A)), expected_text="Type[A]")
    reveal_type(instantiate(just(A)()), expected_text="Type[A]")  # instance of Just


@dataclass
class SomeDataClass:
    pass


def f6():
    some_dataclass = SomeDataClass()

    out1 = instantiate(SomeDataClass)
    out2 = instantiate(some_dataclass)


def f7():
    # get_target(Type[Builds[T]]) -> T
    reveal_type(get_target(builds(str)), expected_text="Type[str]")
    reveal_type(get_target(builds(str, zen_partial=False)), expected_text="Type[str]")
    reveal_type(get_target(builds(str, zen_partial=True)), expected_text="Type[str]")
    reveal_type(get_target(just(str)), expected_text="Type[str]")

    # get_target(Builds[Callable[...]]) -> Callable[...]
    reveal_type(get_target(builds(f)), expected_text="(x: int) -> int")
    reveal_type(
        get_target(builds(f, zen_partial=False)), expected_text="(x: int) -> int"
    )
    reveal_type(
        get_target(builds(f, zen_partial=True)), expected_text="(x: int) -> int"
    )
    reveal_type(get_target(just(f)), expected_text="(x: int) -> int")

    reveal_type(get_target(builds(str)()), expected_text="Type[str]")
    reveal_type(get_target(builds(str, zen_partial=False)()), expected_text="Type[str]")
    reveal_type(get_target(builds(str, zen_partial=True)()), expected_text="Type[str]")
    reveal_type(get_target(just(str)()), expected_text="Type[str]")


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
    reveal_type(builds(str, zen_wrappers=f), expected_text="Type[Builds[Type[str]]]")
    reveal_type(builds(str, zen_wrappers=J), expected_text="Type[Builds[Type[str]]]")
    reveal_type(builds(str, zen_wrappers=B), expected_text="Type[Builds[Type[str]]]")
    reveal_type(builds(str, zen_wrappers=PB), expected_text="Type[Builds[Type[str]]]")
    reveal_type(
        builds(str, zen_wrappers=(None,), expected_text="Type[Builds[Type[str]]]")
    )

    reveal_type(
        builds(str, zen_wrappers=(f, J, B, PB, None)),
        expected_text="Type[Builds[Type[str]]]",
    )

    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=f),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=J),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=B),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=PB),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(None,)),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(f, J, B, PB, None)),
        expected_text="Type[PartialBuilds[Type[str]]]",
    )

    # should fail
    builds(str, zen_wrappers=(2.0, 1))  # type: ignore
    builds(str, zen_wrappers=False)  # type: ignore


def custom_builds_fn():
    _builds = make_custom_builds_fn()

    reveal_type(_builds(int), expected_text="Type[Builds[Type[int]]]")
    reveal_type(
        _builds(int, zen_partial=True), expected_text="Type[PartialBuilds[Type[int]]]"
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

    reveal_type(
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
        ),
        expected_text="Type[DataClass]",
    )
    reveal_type(
        make_config(
            ZenField(name="a", default={M}),
            ZenField(name="b", default={1: M}),
            ZenField(name="c", default=[2.0 + 1j]),
            d=ZenField(default=(1, "hi", 2.0, 1j, set(), M, Path.cwd())),
            e=ZenField(default=f),
        ),
        expected_text="Type[DataClass]",
    )

    reveal_type(
        builds(
            dict,
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(dict), Path.cwd()),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
        ),
        expected_text="Type[Builds[Type[dict[Unknown, Unknown]]]]",
    )

    reveal_type(
        builds(
            dict,
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(dict), Path.cwd()),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
            zen_partial=True,
        ),
        expected_text="Type[PartialBuilds[Type[dict[Unknown, Unknown]]]]",
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
    make_config(a=M())  # type: ignore
    make_config(a=(1, M()))  # type: ignore
    make_config(a=[1, M()])  # type: ignore
    builds(dict, a=M())  # type: ignore

    # This should fail, but doesn't. Seems like a pyright bug
    # make_config(a={"a": M()})  # type: ignore

    # The following *should* be invalid, but we are limited
    # by mutable invariants being generic
    # make_config(a={1j: 1})
    # make_config(a={M: 1})
    # make_config(a={ADataclass: 1})


def check_inheritance():
    P1 = make_config(x=1)
    P2 = builds(dict)

    @dataclass
    class P3:
        pass

    reveal_type(make_config(x=1, bases=(P1, P2, P3)), expected_text="Type[DataClass]")
    reveal_type(
        builds(int, bases=(P1, P2, P3)), expected_text="Type[Builds[Type[int]]]"
    )

    # should fail
    make_config(x=1, bases=(lambda x: x,))  # type: ignore
    make_config(x=1, bases=(None,))  # type: ignore
    make_config(x=1, bases=(A,))  # type: ignore


def make_hydra_partial(x: T) -> HydraPartialBuilds[Type[T]]:
    ...


def check_HydraPartialBuilds():
    cfg = make_hydra_partial(int)
    reveal_type(instantiate(cfg), expected_text="Partial[int]")


def check_partial_protocol():
    x: Partial[int]
    x = partial(int)
    x = partial(str)  # type: ignore


def check_partiald_target():
    reveal_type(builds(partial(int)), expected_text="Type[Builds[partial[int]]]")
    reveal_type(
        builds(partial(int), zen_partial=True),
        expected_text="Type[PartialBuilds[partial[int]]]",
    )
    a = builds(partial(int))
    reveal_type(instantiate(a), expected_text="int")

    b = builds(partial(int), zen_partial=True)
    reveal_type(instantiate(b), expected_text="Partial[int]")


def check_target_annotation():
    builds(int)
    builds(print)
    builds(partial(int))

    # should fail:
    builds()  # type: ignore
    builds(1)  # type: ignore
    builds(None)  # type: ignore


def check_protocols():
    reveal_type(builds(int)._target_, expected_text="str")
    reveal_type(builds(int)()._target_, expected_text="str")

    PBuilds = builds(int, zen_partial=True)
    reveal_type(
        PBuilds._target_, expected_text="Literal['hydra_zen.funcs.zen_processing']"
    )
    reveal_type(
        PBuilds()._target_, expected_text="Literal['hydra_zen.funcs.zen_processing']"
    )

    reveal_type(PBuilds._zen_target, expected_text="str")
    reveal_type(PBuilds()._zen_target, expected_text="str")

    reveal_type(PBuilds._zen_partial, expected_text="Literal[True]")
    reveal_type(PBuilds()._zen_partial, expected_text="Literal[True]")

    Just = just(int)
    reveal_type(Just._target_, expected_text="Literal['hydra_zen.funcs.get_obj']")
    reveal_type(Just()._target_, expected_text="Literal['hydra_zen.funcs.get_obj']")
