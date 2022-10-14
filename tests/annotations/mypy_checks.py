# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing_extensions import assert_type

from hydra_zen import builds, instantiate, make_custom_builds_fn


def check_builds() -> None:
    class A:
        ...

    def f(x: int) -> str:
        ...

    assert_type(instantiate(builds(A)), A)
    assert_type(instantiate(builds(A, populate_full_signature=True)), A)
    assert_type(instantiate(builds(A, zen_partial=True))(), A)

    assert_type(instantiate(builds(f)), str)
    assert_type(instantiate(builds(f, populate_full_signature=True)), str)
    assert_type(instantiate(builds(f, zen_partial=True))(), str)

    builds(f, populate_full_signature=True)(y=2)  # type: ignore [call-arg]


def check_make_custom_builds() -> None:
    class A:
        ...

    def f(x: int) -> str:
        ...

    builds_ = make_custom_builds_fn()
    partial_builds = make_custom_builds_fn(zen_partial=True)
    full_builds = make_custom_builds_fn(populate_full_signature=True)

    assert_type(instantiate(builds_(A)), A)
    assert_type(instantiate(full_builds(A)), A)
    assert_type(instantiate(partial_builds(A))(), A)

    assert_type(instantiate(builds_(f)), str)
    assert_type(instantiate(full_builds(f)), str)
    assert_type(instantiate(partial_builds(f))(), str)

    full_builds(f)(y=2)  # type: ignore [call-arg]


# def check_just() -> None:
#     from hydra_zen import just

#     class A:
#         ...

#     # fails due to: https://github.com/python/mypy/issues/13623
#     assert_type(instantiate(just(A)), Type[A])


## hydrated_dataclass not yet supported: https://github.com/python/mypy/issues/12840
# def check_hydrated_dataclass() -> None:
#     from hydra_zen import hydrated_dataclass

#     @hydrated_dataclass(dict)
#     class A:
#         x: int

#     assert_type(A(1).x, int)
