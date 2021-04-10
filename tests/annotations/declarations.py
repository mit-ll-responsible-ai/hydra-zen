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
conf_f_partial: PartialBuilds[f_sig] = builds(f, hydra_partial=True)
partial_out_f: Partial[f_sig] = instantiate(conf_f_partial)

# test builds(..., hydra_partial=False)
conf_A: Builds[Type[A]] = builds(A, hydra_partial=False)
should_be_a_again: A = instantiate(conf_A)

conf_f: Builds[f_sig] = builds(f, hydra_partial=False)
should_be_int: int = instantiate(conf_f)

# test builds(...)
conf_A: Builds[Type[A]] = builds(A)
should_be_a_again: A = instantiate(conf_A)

conf_f: Builds[f_sig] = builds(f)
should_be_int: int = instantiate(conf_f)

# test just(...)
conf_just_f: Just[f_sig] = just(f)
conf_just_A: Just[Type[A]] = just(A)
