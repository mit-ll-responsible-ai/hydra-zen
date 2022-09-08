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
