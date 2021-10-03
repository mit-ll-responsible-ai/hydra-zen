from typing import Tuple

import pytest
from beartype.roar import BeartypeException  # typing: ignore

from hydra_zen import builds, instantiate
from hydra_zen.experimental.third_party.beartype import validates_with_beartype


class A:
    pass


def f(x: A):
    return None


class B:
    def __init__(self, x: A) -> None:
        pass


def test_bear_basics():
    C1 = builds(f, builds(A), zen_wrappers=validates_with_beartype)
    C2 = builds(B, builds(A), zen_wrappers=validates_with_beartype)
    instantiate(C1)
    assert isinstance(instantiate(C2), B)


class F:
    pass


class G:
    pass


def tuple_of_classes(x: Tuple[F, G]):
    return x


def test_beartype_with_arbitrary_allowed_types():
    instantiate(builds(tuple_of_classes, [builds(F), builds(F)]))  # should pass

    with pytest.raises(BeartypeException):
        instantiate(
            builds(
                tuple_of_classes,
                [builds(F), builds(F)],
                zen_wrappers=validates_with_beartype,
                hydra_convert="all",
            )
        )

    f, g = instantiate(
        builds(
            tuple_of_classes,
            [builds(F), builds(G)],
            zen_wrappers=validates_with_beartype,
            hydra_convert="all",
        )
    )
    assert isinstance(f, F)
    assert isinstance(g, G)
