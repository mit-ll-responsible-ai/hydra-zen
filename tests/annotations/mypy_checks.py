# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from typing_extensions import assert_type

from hydra_zen import builds, instantiate, just, make_custom_builds_fn, store
from hydra_zen.wrapper import ZenStore


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


def check_store() -> None:
    @store
    def f(x: int, y: int) -> str:
        ...

    @store(name="hi")
    def f2(x: int, y: int) -> str:
        ...

    # reveal_type(f)
    # reveal_type(f2)

    # reveal_type(store(f))
    # reveal_type(store(f, name="bye"))
    # reveal_type(store(name="bye")(f))

    apple_store = store(group="apple")
    assert_type(apple_store, ZenStore)

    @apple_store
    def a1(x: int) -> bool:
        ...

    @apple_store(name="hello")
    def a2(x: int) -> bool:
        ...

    assert_type(a2(1), bool)
    assert_type(store()(a1)(1), bool)
    assert_type(store()(a1)("a"), bool)  # type: ignore [arg-type]
    apple_store(name="bye")
    apple_store(name=22)  # type: ignore [call-overload]

    # reveal_type(apple_store(a1))
    # reveal_type(apple_store(a1, name="bye"))
    # reveal_type(apple_store(name="bye")(a1))

    @store(f)  # type: ignore [arg-type, call-arg]
    def bad(x: int, y: int) -> str:
        ...

    # checking that store type-checks against to_config
    # mypy isn't as good as pyright here

    # store(1)  # false negative
    # store()(1, to_config=builds)  # false negative
    store(1, to_config=just)
    store()(1, to_config=just)


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
