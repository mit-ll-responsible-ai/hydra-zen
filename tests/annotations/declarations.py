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
from typing import Any, Callable, List, Tuple, Type, TypeVar, Union

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
from hydra_zen.typing import (
    Builds,
    HydraPartialBuilds,
    Partial,
    PartialBuilds,
    ZenPartialBuilds,
)
from hydra_zen.typing._builds_overloads import full_builds, partial_builds
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
        expected_text="Type[ZenPartialBuilds[Type[dict[Unknown, Unknown]]]] | Type[HydraPartialBuilds[Type[dict[Unknown, Unknown]]]]",
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


def check_full_builds():
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


def check_partial_builds():
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

    Parent = make_config(x=1)

    # specifying parent should produce default `builds`
    builds2 = make_custom_builds_fn(
        populate_full_signature=True, builds_bases=(Parent,)
    )

    Conf2 = builds2(f)

    reveal_type(
        Conf2,
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

    Parent = make_config(x=1)

    partial_builds3 = make_custom_builds_fn(
        zen_partial=True, builds_bases=(Parent,), populate_full_signature=True
    )

    Conf3 = partial_builds3(f)

    reveal_type(
        Conf3,
        expected_text="Type[ZenPartialBuilds[(x: int, y: str, z: bool = False) -> int]] | Type[HydraPartialBuilds[(x: int, y: str, z: bool = False) -> int]]",
    )


def check_protocol_compatibility():
    def f_builds(x: Type[Builds]):
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
