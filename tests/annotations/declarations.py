# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing import Callable, Tuple, Type

from hydra_zen import builds, instantiate, just
from hydra_zen.typing import Builds, Just, Partial, PartialBuilds


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


# test type behaviors

# test builds(..., hydra_partial=True)
conf_a_partial = builds(
    A, hydra_partial=True
)  # there is something really weird where annotating : PartialBuilds[Type[A]] breaks this..
partial_out: Partial[A] = instantiate(conf_a_partial)
out_a: A = partial_out()

f_sig = Callable[[int], int]
conf_f_partial: Type[PartialBuilds[f_sig]] = builds(f, hydra_partial=True)
conf_f_partial_instance = conf_f_partial()
partial_out_f: Partial[int] = instantiate(conf_f_partial)
partial_out_f_via_instance: Partial[int] = instantiate(conf_f_partial_instance)

# test builds(..., hydra_partial=False)
conf_A_1: Type[Builds[Type[A]]] = builds(A, hydra_partial=False)
should_be_a_again: A = instantiate(conf_A_1)

conf_f_1: Type[Builds[f_sig]] = builds(f, hydra_partial=False)
should_be_int: int = instantiate(conf_f_1)

# test builds(...)
conf_A_2: Type[Builds[Type[A]]] = builds(A)
should_be_a_again_again: A = instantiate(conf_A_2)

conf_a_instance: Builds[Type[A]] = conf_A_2()
should_be_a_via_instance: A = instantiate(conf_a_instance)

conf_f_2: Type[Builds[f_sig]] = builds(f)
should_be_int_again: int = instantiate(conf_f_2)

# test just(...)
conf_just_f: Type[Just[f_sig]] = just(f)
conf_just_A: Type[Just[Type[A]]] = just(A)
