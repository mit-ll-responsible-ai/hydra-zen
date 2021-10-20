# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing import Tuple

from hydra_zen import builds, hydrated_dataclass, instantiate, just


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


def g(x: int, y: float) -> str:
    ...


def behaviors():
    # test type behaviors

    # test builds(..., zen_partial=True)
    conf_a_partial = builds(A, zen_partial=True)
    partial_out = instantiate(conf_a_partial)
    should_be_a = partial_out()
    out: Tuple[int, str] = should_be_a.x

    conf_a_partial_instance = builds(A, zen_partial=True)()
    partial_out_2 = instantiate(conf_a_partial_instance)
    should_be_a_2 = partial_out_2()
    out_2: Tuple[int, str] = should_be_a_2.x

    conf_f_partial = builds(f, zen_partial=True)
    partial_out_f = instantiate(conf_f_partial)
    should_be_int_output_of_f: int = partial_out_f()

    conf_f_partial_instance = builds(f, zen_partial=True)()
    partial_out_f_2 = instantiate(conf_f_partial_instance)
    should_be_int_output_of_f_2: int = partial_out_f_2()

    # test builds(..., zen_partial=False)
    conf_A = builds(A)
    should_be_a_again = instantiate(conf_A)
    out2: Tuple[int, str] = should_be_a_again.x

    conf_f = builds(f)
    conf_f_instance = conf_f()
    should_be_int: int = instantiate(conf_f)
    should_be_int_2: int = instantiate(conf_f_instance)

    # test just(...)
    conf_just_f = just(f)
    just_f = instantiate(conf_just_f)
    yet_another_int: int = just_f(10)

    conf_just_f_instance = just(f)()
    just_f_2 = instantiate(conf_just_f_instance)
    yet_another_int_2: int = just_f_2(10)

    conf_just_A = just(A)
    just_A = instantiate(conf_just_A)
    instance_of_a = just_A()
    out3: Tuple[int, str] = instance_of_a.x

    conf_just_A_instance = just(A)()
    just_A_2 = instantiate(conf_just_A_instance)
    instance_of_a = just_A_2()
    out3: Tuple[int, str] = instance_of_a.x

    @hydrated_dataclass(A)
    class B:
        x: int

    # Check that @hydrated_dataclass reveals init/attr info
    b = B(x=2)
    b.x = 3

    # Check that `Builds` constructor can take arguments
    X = builds(dict, a=1)
    y = X(a=10)

    PartialBuild_g = builds(g, x=1, zen_partial=True)
    partial_g = instantiate(PartialBuild_g)
    g_out: str = partial_g(y=10)
