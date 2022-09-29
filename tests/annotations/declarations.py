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
from typing import Any, Callable, List, Mapping, Optional, Tuple, Type, TypeVar, Union

from omegaconf import MISSING, DictConfig, ListConfig
from typing_extensions import Literal, assert_type

from hydra_zen import (
    ZenField,
    builds,
    get_target,
    instantiate,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
    zen,
)
from hydra_zen.structured_configs._value_conversion import ConfigComplex, ConfigPath
from hydra_zen.typing import (
    Builds,
    HydraPartialBuilds,
    Partial,
    PartialBuilds,
    SupportedPrimitive,
    ZenPartialBuilds,
    ZenWrappers,
)
from hydra_zen.typing._builds_overloads import FullBuilds, PBuilds, StdBuilds
from hydra_zen.typing._implementations import DataClass_, HydraPartialBuilds
from hydra_zen.wrapper import Zen

T = TypeVar("T")


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


# test type behaviors
def check_partial_builds_on_class():
    reveal_type(
        builds(A, zen_partial=True),
        expected_text="Type[ZenPartialBuilds[Type[A]]] | Type[HydraPartialBuilds[Type[A]]]",
    )
    conf_a_partial = builds(A, zen_partial=True)
    reveal_type(instantiate(conf_a_partial), expected_text="Partial[A]")
    reveal_type(instantiate(conf_a_partial)(), expected_text="A")


f_sig = Callable[[int], int]


def check_partial_builds_on_function():
    reveal_type(
        builds(f, zen_partial=True),
        expected_text="Type[ZenPartialBuilds[(x: int) -> int]] | Type[HydraPartialBuilds[(x: int) -> int]]",
    )

    conf_f_partial = builds(f, zen_partial=True)

    reveal_type(
        conf_f_partial(),
        expected_text="ZenPartialBuilds[(x: int) -> int] | HydraPartialBuilds[(x: int) -> int]",
    )

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


def check_just():
    def f(x: int) -> int:
        return x

    class A:
        ...

    # test just(...)
    reveal_type(just(f), expected_text="Type[Just[(x: int) -> int]]")
    reveal_type(just(A), expected_text="Type[Just[Type[A]]]")
    reveal_type(instantiate(just(f)), expected_text="(x: int) -> int")
    reveal_type(instantiate(just(A)), expected_text="Type[A]")
    reveal_type(instantiate(just(A)()), expected_text="Type[A]")  # instance of Just

    reveal_type(just(1), expected_text="int")
    reveal_type(just("hi"), expected_text="str")
    reveal_type(just(b"1234"), expected_text="bytes")
    reveal_type(just(1 + 2j), expected_text="ConfigComplex")
    reveal_type(just(Path.home()), expected_text="Path")
    reveal_type(just(partial(f, 1)), expected_text="Type[Just[partial[int]]]")
    reveal_type(just(set([1, 2, 3])), expected_text="Builds[Type[set[int]]]")
    reveal_type(just(range(10)), expected_text="Builds[Type[range]]")

    partiald_f = instantiate(just(partial(f, 1)))
    reveal_type(partiald_f, expected_text="partial[int]")
    reveal_type(partiald_f(), expected_text="int")

    # test dataclass conversion
    @dataclass
    class B:
        ...

    reveal_type(just(B), expected_text="Type[Just[Type[B]]]")
    reveal_type(just(B()), expected_text="Type[Builds[Type[B]]]")
    reveal_type(just(B(), zen_convert={"dataclass": False}), expected_text="Any")


@dataclass
class SomeDataClass:
    pass


def f6():
    some_dataclass = SomeDataClass()

    reveal_type(instantiate(SomeDataClass), expected_text="Any")
    reveal_type(instantiate(some_dataclass), expected_text="Any")


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

    reveal_type(A().x, expected_text="List[int]")


def zen_wrappers():
    def f(obj: Any):
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
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=J),
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=B),
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=PB),
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(None,)),
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )
    reveal_type(
        builds(str, zen_partial=True, zen_wrappers=(f, J, B, PB, None)),
        expected_text="Type[ZenPartialBuilds[Type[str]]] | Type[HydraPartialBuilds[Type[str]]]",
    )

    # should fail
    builds(str, zen_wrappers=(2.0, 1))  # type: ignore
    builds(str, zen_wrappers=False)  # type: ignore


def custom_builds_fn():
    _builds = make_custom_builds_fn()

    reveal_type(_builds(int), expected_text="Type[Builds[Type[int]]]")
    reveal_type(
        _builds(int, zen_partial=True),
        expected_text="Type[ZenPartialBuilds[Type[int]]] | Type[HydraPartialBuilds[Type[int]]]",
    )


def supported_primitives():
    class M:
        pass

    def f(*args: Any):
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
                {1, 2},
                M,
                ADataclass,
                builds(int),
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
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(int), Path.cwd()),
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
            a=(1, "hi", 2.0, 1j, set(), M, ADataclass, builds(int), Path.cwd()),
            b={M},
            c={1: M},
            d=[2.0 + 1j],
            e=f,
            zen_partial=True,
        ),
        expected_text="Type[ZenPartialBuilds[Type[dict[Unknown, Unknown]]]] | Type[HydraPartialBuilds[Type[dict[Unknown, Unknown]]]]",
    )

    # check lists
    make_config(a=[], b=[1], c=[[1]], d=[[[M]]])

    # check dicts
    make_config(
        a={}, b={1: 1}, c=[{1: 1}], d={1: {"a": "a"}}, e={"a": 1j}, f={"a": [1j]}
    )

    builds(
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
        deque([1, 2]),
        Counter(),
        [deque([1, 2]), Counter({1: 1}), 1j],
        (deque([1, 2]), Counter(), 1j),
        range(1, 10, 2),
        odict,
        olist,
    )

    a_list = [1, 2, [1, 2]]
    a_dict = {"a": [1, 2, [1, 2]]}
    a_set = {1, 2.0, (1, 2)}

    # make sure we don't hit this issue again
    # https://github.com/microsoft/pyright/issues/2659
    make_config(x=a_list, y=a_dict, z=a_set)

    # The following should be marked as "bad by type-checkers
    make_config(a=M())  # type: ignore
    make_config(a=(1, M()))  # type: ignore
    make_config(a=[1, M()])  # type: ignore
    make_config(a={"a": M()})  # type: ignore
    builds(dict, a=M())  # type: ignore

    # The following *should* be invalid, but we are limited
    # by mutable invariants being generic
    # make_config(a={1j: 1})
    # make_config(a={M: 1})
    # make_config(a={ADataclass: 1})


def check_zen_field():
    ZenField(int)
    ZenField(Callable[..., Any])
    ZenField(List[int])

    ZenField(1.0)  # type: ignore

    # this is a known limitation
    # TODO: open pyright issue
    ZenField(Union[int, str])  # type: ignore


def check_base_annotations():
    P1 = make_config(x=1)
    P2 = builds(int)

    @dataclass
    class P3:
        pass

    def f(x: int):
        pass

    P4 = builds(f, populate_full_signature=True)

    reveal_type(
        make_config(x=1, bases=(P1, P2, P3, P4)), expected_text="Type[DataClass]"
    )
    reveal_type(
        builds(int, bases=(P1, P2, P3, P4)), expected_text="Type[Builds[Type[int]]]"
    )

    # should fail
    make_config(x=1, bases=(lambda x: x,))  # type: ignore
    make_config(x=1, bases=(None,))  # type: ignore
    make_config(x=1, bases=(A,))  # type: ignore

    # should fail
    make_custom_builds_fn(builds_bases=(lambda x: x,))  # type: ignore
    make_custom_builds_fn(builds_bases=(None,))  # type: ignore
    make_custom_builds_fn(builds_bases=(A,))  # type: ignore


def make_hydra_partial(x: T) -> HydraPartialBuilds[T]:
    ...


def check_HydraPartialBuilds():
    cfg = make_hydra_partial(int)
    reveal_type(instantiate(cfg), expected_text="Partial[int]")


def check_partial_protocol():
    x: Partial[int]
    x = partial(int)
    x = partial(str)  # type: ignore
    assert x


def check_partiald_target():
    reveal_type(builds(partial(int)), expected_text="Type[Builds[partial[int]]]")
    reveal_type(
        builds(partial(int), zen_partial=True),
        expected_text="Type[ZenPartialBuilds[partial[int]]] | Type[HydraPartialBuilds[partial[int]]]",
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
    def f() -> Type[ZenPartialBuilds[Type[int]]]:
        ...

    def g() -> Type[HydraPartialBuilds[Type[int]]]:
        ...

    reveal_type(builds(int)._target_, expected_text="str")
    reveal_type(builds(int)()._target_, expected_text="str")

    PBuilds = f()

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

    HPBuilds = g()
    reveal_type(HPBuilds._partial_, expected_text="Literal[True]")
    reveal_type(HPBuilds()._partial_, expected_text="Literal[True]")

    Just = just(int)
    reveal_type(Just._target_, expected_text="Literal['hydra_zen.funcs.get_obj']")
    reveal_type(Just()._target_, expected_text="Literal['hydra_zen.funcs.get_obj']")


def check_populate_full_sig():
    class C:
        def __init__(self) -> None:
            pass

    def f(x: int, y: str, z: bool = False):
        return C()

    # should be able to specify arguments other than `populate_full_signature=True`
    Conf_f = builds(
        f,
        populate_full_signature=True,
        zen_wrappers=[None],
        zen_meta=dict(a=1),
        hydra_convert="all",
    )

    # The following should get flagged by type-checkers

    Conf_f()  # type: ignore
    Conf_f(not_a_arg=1)  # type: ignore
    Conf_f("not int", "a string")  # type: ignore
    Conf_f(x="not int", y="a string")  # type: ignore
    Conf_f(x=1, y=2)  # type: ignore
    Conf_f(x=1, y="a string", z="not a bool")  # type: ignore

    # The following should be ok
    reveal_type(
        Conf_f(1, "hi"),
        expected_text="BuildsWithSig[Type[C], (x: int, y: str, z: bool = ...)]",
    )
    reveal_type(
        Conf_f(1, "hi", True),
        expected_text="BuildsWithSig[Type[C], (x: int, y: str, z: bool = ...)]",
    )
    reveal_type(
        Conf_f(1, y="hi"),
        expected_text="BuildsWithSig[Type[C], (x: int, y: str, z: bool = ...)]",
    )
    reveal_type(
        Conf_f(x=1, y="hi", z=False),
        expected_text="BuildsWithSig[Type[C], (x: int, y: str, z: bool = ...)]",
    )

    # check instantiation
    reveal_type(instantiate(Conf_f), expected_text="C")

    conf1 = Conf_f(1, "hi")
    reveal_type(instantiate(conf1), expected_text="C")

    Conf_C = builds(C, populate_full_signature=True)

    # Should be flagged by type-checker
    Conf_C(not_a_arg=1)  # type: ignore

    reveal_type(instantiate(Conf_C), expected_text="C")
    reveal_type(instantiate(Conf_C()), expected_text="C")

    # specifying `zen_partial=True` should disable sig-reflection
    Conf_f_partial = builds(f, populate_full_signature=True, zen_partial=True)
    conf2 = Conf_f_partial(not_a_valid_arg=1)  # should be ok
    reveal_type(
        conf2,
        expected_text="ZenPartialBuilds[(x: int, y: str, z: bool = False) -> C] | HydraPartialBuilds[(x: int, y: str, z: bool = False) -> C]",
    )

    # specifying `populate_full_signature=False` should disable sig-reflection
    Conf_f_not_full = builds(f, populate_full_signature=False)
    conf3 = Conf_f_not_full(not_a_valid_arg=1)  # should be ok
    reveal_type(conf3, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # not specifying `populate_full_signature` should disable sig-reflection
    Conf_f_not_full2 = builds(f)
    conf4 = Conf_f_not_full2(not_a_valid_arg=1)  # should be ok
    reveal_type(conf4, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any *args directly in `builds` should distable sig-reflection
    Conf_f_with_args = builds(f, 1, populate_full_signature=True)
    conf5 = Conf_f_with_args()  # should be ok
    reveal_type(conf5, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any **kwargs directly in `builds` should distable sig-reflection
    Conf_f_with_kwargs = builds(f, x=1, populate_full_signature=True)
    conf6 = Conf_f_with_kwargs()  # should be ok
    reveal_type(conf6, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any bases in `builds` should distable sig-reflection
    Parent = make_config(x=1)
    Conf_f_with_base = builds(f, populate_full_signature=True, builds_bases=(Parent,))
    conf7 = Conf_f_with_base()  # should be ok
    reveal_type(conf7, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")


def check_full_builds(full_builds: FullBuilds):
    def f(x: int, y: str, z: bool = False):
        return 1

    # type-checker should see default: `populate_full_signature=True`
    Conf_f = full_builds(f)
    Conf_f()  # type: ignore

    # explicitly specifying `populate_full_signature=True` should produce
    # same behaviore
    Conf_f2 = full_builds(f, populate_full_signature=True)
    Conf_f2()  # type: ignore

    Conf_f3 = full_builds(f, zen_partial=True)
    reveal_type(
        Conf_f3,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]]",
    )
    Conf_f3()

    class C:
        def __init__(self) -> None:
            pass

    def g(x: int, y: str, z: bool = False):
        return C()

    # specifying `populate_full_signature=False` should disable sig-reflection
    Conf_f_not_full = full_builds(g, populate_full_signature=False)
    conf3 = Conf_f_not_full(not_a_valid_arg=1)  # should be ok
    reveal_type(conf3, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any *args directly in `builds` should distable sig-reflection
    Conf_f_with_args = full_builds(g, 1, populate_full_signature=True)
    conf5 = Conf_f_with_args()  # should be ok
    reveal_type(conf5, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any **kwargs directly in `builds` should distable sig-reflection
    Conf_f_with_kwargs = full_builds(g, x=1, populate_full_signature=True)
    conf6 = Conf_f_with_kwargs()  # should be ok
    reveal_type(conf6, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")

    # Providing any bases in `builds` should distable sig-reflection
    Parent = make_config(x=1)
    Conf_f_with_base = full_builds(
        g, populate_full_signature=True, builds_bases=(Parent,)
    )
    conf7 = Conf_f_with_base()  # should be ok
    reveal_type(conf7, expected_text="Builds[(x: int, y: str, z: bool = False) -> C]")


def check_partial_builds(partial_builds: PBuilds):
    def f(x: int, y: str, z: bool = False):
        return 1

    # type-checker should see default: `populate_full_signature=True`
    Conf_f = partial_builds(f)
    reveal_type(
        Conf_f,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]]",
    )

    # type-checker should see default: `populate_full_signature=True`
    Conf_f2 = partial_builds(f, zen_partial=True)
    reveal_type(
        Conf_f2,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> Literal[1]]]",
    )

    # signature should be required
    Conf_f3 = partial_builds(f, zen_partial=False, populate_full_signature=True)
    Conf_f3()  # type: ignore


def check_make_custom_builds_no_args():
    def f(x: int, y: str, z: bool = False):
        return 1

    builds_ = make_custom_builds_fn()
    Conf = builds_(f)
    Conf()  # should be OK

    reveal_type(
        Conf,
        expected_text="Type[Builds[(x: int, y: str, z: bool = False) -> Literal[1]]]",
    )


def check_make_custom_builds_pop_sig():
    def f(x: int, y: str, z: bool = False):
        return 1

    full_builds = make_custom_builds_fn(populate_full_signature=True)

    Conf = full_builds(f)
    Conf()  # type: ignore

    reveal_type(
        Conf,
        expected_text="Type[BuildsWithSig[Type[int], (x: int, y: str, z: bool = ...)]]",
    )


def check_make_custom_builds_partial():
    def f(x: int, y: str, z: bool = False) -> int:
        return 1

    partial_builds = make_custom_builds_fn(zen_partial=True)

    Conf = partial_builds(f)

    reveal_type(
        Conf,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> int]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> int]]",
    )

    partial_builds2 = make_custom_builds_fn(
        zen_partial=True, populate_full_signature=True
    )

    Conf2 = partial_builds2(f)

    reveal_type(
        Conf2,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> int]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> int]]",
    )


def check_protocol_compatibility():
    def f_builds(x: Type[Builds[Any]]):
        pass

    def f_partial(x: Type[PartialBuilds[Any]]):
        pass

    def f(x: int):
        pass

    b = builds(int)
    pb = builds(int, zen_partial=True)
    bs = builds(f, populate_full_signature=True)

    f_builds(b)
    f_builds(pb)
    f_builds(bs)

    f_partial(b)  # type: ignore
    f_partial(pb)
    f_partial(bs)  # type: ignore


def check_targeted_dataclass():
    @dataclass
    class OptimizerConf:
        _target_: str
        _partial_: bool = True

    optimizer_conf = OptimizerConf(_target_="torch.optim.SGD")
    reveal_type(instantiate(optimizer_conf), expected_text="Any")


def check_overloads_arent_too_restrictive():
    def caller(
        zen_partial: Optional[bool],
        zen_wrappers: ZenWrappers[Callable[..., Any]],
        zen_meta: Optional[Mapping[str, SupportedPrimitive]],
        populate_full_signature: bool,
        hydra_recursive: Optional[bool],
        hydra_convert: Optional[Literal["none", "partial", "all"]],
        frozen: bool,
        builds_bases: Tuple[Type[DataClass_], ...],
        dataclass_name: Optional[str],
        fbuilds: FullBuilds = ...,
        pbuilds: PBuilds = ...,
        **kwargs_for_target: SupportedPrimitive
    ):

        bout = builds(
            int,
            zen_partial=zen_partial,
            zen_wrappers=zen_wrappers,
            zen_meta=zen_meta,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            frozen=frozen,
            builds_bases=builds_bases,
            dataclass_name=dataclass_name,
            **kwargs_for_target
        )

        reveal_type(
            bout,
            expected_text="Type[Builds[Type[int]]] | Type[ZenPartialBuilds[Type[int]]] | Type[HydraPartialBuilds[Type[int]]] | Type[BuildsWithSig[Unknown, (...)]]",
        )

        fout = fbuilds(
            int,
            zen_partial=zen_partial,
            zen_wrappers=zen_wrappers,
            zen_meta=zen_meta,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            frozen=frozen,
            builds_bases=builds_bases,
            dataclass_name=dataclass_name,
            **kwargs_for_target
        )

        reveal_type(
            fout,
            expected_text="Type[Builds[Type[int]]] | Type[ZenPartialBuilds[Type[int]]] | Type[HydraPartialBuilds[Type[int]]] | Type[BuildsWithSig[Unknown, (...)]]",
        )

        pout = pbuilds(
            int,
            zen_partial=zen_partial,
            zen_wrappers=zen_wrappers,
            zen_meta=zen_meta,
            populate_full_signature=populate_full_signature,
            hydra_recursive=hydra_recursive,
            hydra_convert=hydra_convert,
            frozen=frozen,
            builds_bases=builds_bases,
            dataclass_name=dataclass_name,
            **kwargs_for_target
        )

        reveal_type(
            pout,
            expected_text="Type[Builds[Type[int]]] | Type[ZenPartialBuilds[Type[int]]] | Type[HydraPartialBuilds[Type[int]]] | Type[BuildsWithSig[Unknown, (...)]]",
        )

    assert caller


def check_hydra_target_pos_only(partial_builds: PBuilds, full_builds: FullBuilds):
    builds(hydra_target=int)  # type: ignore
    builds(__hydra_target=int)  # type: ignore

    partial_builds(hydra_target=int)  # type: ignore
    partial_builds(__hydra_target=int)  # type: ignore

    full_builds(hydra_target=int)  # type: ignore
    full_builds(__hydra_target=int)  # type: ignore


# the following tests caught regressions in builds' overloads


def check_partial_narrowing_builds(x: bool, partial_: Optional[bool]):
    def f(x: int):
        return 1

    maybe_full_sig = builds(f, zen_partial=partial_, populate_full_signature=x)
    # could have full-sig; should raise
    maybe_full_sig()  # type: ignore

    not_full_sig = builds(f, zen_partial=partial_, populate_full_signature=False)
    # can't have full sig,
    not_full_sig()

    not_full_sig2 = builds(f, zen_partial=partial_)
    # can't have full-sig
    not_full_sig2()


def check_partial_narrowing_std_and_partial(
    builds_fn: Union[StdBuilds, PBuilds], x: bool, partial_: Optional[bool]
):
    def f(x: int):
        return 1

    maybe_full_sig = builds_fn(f, zen_partial=partial_, populate_full_signature=x)
    # could have full-sig; should raise
    maybe_full_sig()  # type: ignore

    not_full_sig = builds_fn(f, zen_partial=partial_, populate_full_signature=False)
    # can't have full sig,
    not_full_sig()

    not_full_sig2 = builds_fn(f, zen_partial=partial_)
    # can't have full-sig
    not_full_sig2()


def check_partial_narrowing_full(
    builds_fn: FullBuilds, x: bool, partial_: Optional[bool]
):
    def f(x: int):
        return 1

    maybe_full_sig = builds_fn(f, zen_partial=partial_, populate_full_signature=x)
    # could have full-sig; should raise
    maybe_full_sig()  # type: ignore

    not_full_sig = builds_fn(f, zen_partial=partial_, populate_full_signature=False)
    # can't have full sig,
    not_full_sig()

    yes_full_sig = builds_fn(f, zen_partial=partial_)
    # can't have full-sig
    yes_full_sig()  # type: ignore


def check_make_custom_builds_overloads(boolean: bool, optional_boolean: Optional[bool]):

    # partial = False, pop-sig = False
    reveal_type(make_custom_builds_fn(zen_partial=False), expected_text="StdBuilds")

    reveal_type(
        make_custom_builds_fn(zen_partial=False, populate_full_signature=False),
        expected_text="StdBuilds",
    )

    # Returns `PBuilds`
    reveal_type(
        make_custom_builds_fn(zen_partial=True, populate_full_signature=boolean),
        expected_text="PBuilds",
    )
    reveal_type(make_custom_builds_fn(zen_partial=True), expected_text="PBuilds")
    reveal_type(
        make_custom_builds_fn(zen_partial=True, populate_full_signature=False),
        expected_text="PBuilds",
    )

    # Returns `PBuilds | StdBuilds``
    reveal_type(
        make_custom_builds_fn(zen_partial=False, populate_full_signature=boolean),
        expected_text="FullBuilds | StdBuilds",
    )

    reveal_type(
        make_custom_builds_fn(zen_partial=optional_boolean),
        expected_text="PBuilds | StdBuilds",
    )

    reveal_type(
        make_custom_builds_fn(
            populate_full_signature=False, zen_partial=optional_boolean
        ),
        expected_text="PBuilds | StdBuilds",
    )

    # Returns `FullBuilds | PBuilds | StdBuilds`
    reveal_type(
        make_custom_builds_fn(
            populate_full_signature=True, zen_partial=optional_boolean
        ),
        expected_text="FullBuilds | PBuilds | StdBuilds",
    )

    reveal_type(
        make_custom_builds_fn(
            populate_full_signature=boolean, zen_partial=optional_boolean
        ),
        expected_text="FullBuilds | PBuilds | StdBuilds",
    )


def check_instantiate_overrides(
    any_: Any, complex_: ConfigComplex, path_: ConfigPath, dataclass_: DataClass_
):
    reveal_type(instantiate(any_), expected_text="Any")
    reveal_type(instantiate(complex_), expected_text="complex")
    reveal_type(instantiate(path_), expected_text="Path")
    reveal_type(instantiate(dataclass_), expected_text="Any")


def check_hydra_defaults(
    partial_builds: PBuilds, full_builds: FullBuilds, std_builds: StdBuilds
):

    builds(int, hydra_defaults=["_self_", {"a": "b"}])
    partial_builds(int, hydra_defaults=["_self_", {"a": "b"}])
    full_builds(int, hydra_defaults=["_self_", {"a": "b"}])
    std_builds(int, hydra_defaults=["_self_", {"a": "b"}])
    make_config(hydra_defaults=["_self_", {"a": "b"}])

    builds(int, hydra_defaults=1)  # type: ignore
    partial_builds(int, hydra_defaults=1)  # type: ignore
    full_builds(int, hydra_defaults=1)  # type: ignore
    std_builds(int, hydra_defaults=1)  # type: ignore
    make_config(hydra_defaults=1)  # type: ignore

    builds(int, hydra_defaults=["_self_", {"a": "b"}, {1: 1}])  # type: ignore
    builds(int, hydra_defaults=["_self_", ["a"]])  # type: ignore
    builds(int, hydra_defaults=["_self_", {("a",): "b"}])  # type: ignore
    builds(int, hydra_defaults={"a": "b"})  # type: ignore
    builds(int, hydra_defaults="_self_")  # type: ignore

    builds(
        int,
        hydra_defaults=[
            "_self_",
            {"a": "b"},
            {"a": None},
            {"a": MISSING},
            {"a": ["b"]},
            DictConfig({"a": "b"}),
            {"a": ListConfig(["a"])},
            make_config(a="a"),
        ],
    )


def check_launch():
    from hydra_zen import launch
    from hydra_zen.typing._implementations import DataClass, InstOrType

    Xonf = make_config()

    def f(x: InstOrType[DataClass]):
        pass

    f(Xonf)
    launch(Xonf, f)


def check_instantiate():
    @dataclass
    class Cfg:
        ...

    assert_type(instantiate(DictConfig({})), Any)
    assert_type(instantiate({}), Any)
    assert_type(instantiate(ListConfig([])), Any)
    assert_type(instantiate([]), Any)
    assert_type(instantiate(Cfg), Any)
    assert_type(instantiate(Cfg()), Any)


def check_zen():
    @zen
    def zen_f(x: int) -> str:
        ...

    assert_type(zen_f({"a": 1}), str)
    assert_type(zen_f(DictConfig({"a": 1})), str)
    assert_type(zen_f([]), str)
    assert_type(zen_f(ListConfig([])), str)
    assert_type(zen_f("some yaml"), str)

    zen_f(1)  # type: ignore
    reveal_type(zen_f.func, expected_text="(x: int) -> str")

    @zen(pre_call=None)
    def zen_f2(x: int) -> str:
        ...

    assert_type(zen_f2({"a": 1}), str)
    assert_type(zen_f2(DictConfig({"a": 1})), str)
    assert_type(zen_f2([]), str)
    assert_type(zen_f2(ListConfig([])), str)
    assert_type(zen_f2("some yaml"), str)

    zen_f2(1)  # type: ignore
    reveal_type(zen_f2.func, expected_text="(x: int) -> str")

    class MyZen(Zen):
        ...

    @zen(ZenWrapper=MyZen)
    def zen_rewrapped(x: int) -> str:
        ...

    reveal_type(zen_rewrapped, expected_text="Zen[(x: int), str]")

    def f(x: int):
        ...

    zen_rewrapped2 = zen(f, ZenWrapper=MyZen)

    reveal_type(zen_rewrapped2, expected_text="Zen[(x: int), None]")

    # valid pre-call
    @zen(pre_call=lambda cfg: None)
    def h1():
        ...

    @zen(pre_call=[lambda cfg: None])
    def h2():
        ...

    @zen(pre_call=zen(lambda x, y: None))
    def h3():
        ...

    # bad pre-call

    @zen(pre_call=1)  # type: ignore
    def g1():
        ...

    @zen(pre_call=lambda x, y: None)  # type: ignore
    def g2():
        ...

    @zen(pre_call=[lambda x, y: None])  # type: ignore
    def g3():
        ...
