from typing import Tuple

from hydra_zen import builds, instantiate, just


class A:
    def __init__(self) -> None:
        self.x: Tuple[int, str] = (1, "hi")


def f(x: int) -> int:
    return x


def requires_A(x: int):
    pass


# test type behaviors

# test builds(..., hydra_partial=True)
conf_a_partial = builds(A, hydra_partial=True)
partial_out = instantiate(conf_a_partial)
should_be_a = partial_out()
out: Tuple[int, str] = should_be_a.x


conf_f_partial = builds(f, hydra_partial=True)
partial_out_f = instantiate(conf_f_partial)
should_be_f = partial_out_f()
should_be_int_output_of_f: int = should_be_f(2)

# test builds(..., hydra_partial=False)
conf_A = builds(A)
should_be_a_again = instantiate(conf_A)
out2: Tuple[int, str] = should_be_a_again.x

conf_f = builds(f)
should_be_int: int = instantiate(conf_f)


# test just(...)
conf_just_f = just(f)
just_f = instantiate(conf_just_f)
yet_another_int: int = just_f(10)

conf_just_A = just(A)
just_A = instantiate(conf_just_A)
instance_of_a = A()
out3: Tuple[int, str] = instance_of_a.x
