from typing import Tuple

import pytest
from pydantic import PositiveInt, ValidationError

from hydra_zen import builds, instantiate
from hydra_zen.experimental.third_party.pydantic import validates_with_pydantic


def needs_pos_int(x: PositiveInt):
    return x


class NeedsPosInt:
    def __init__(self, x: PositiveInt) -> None:
        self.x = x


def test_basic_pydantic_validation_on_func():
    NoVal = builds(needs_pos_int, x=-100)
    WithVal = builds(needs_pos_int, x=-100, zen_wrappers=validates_with_pydantic)
    assert instantiate(NoVal) == -100

    with pytest.raises((ValidationError, TypeError)):
        # Unfortunately the interface for `pydantic.ValidationError` is weird..
        # and breaks Hydra's error handling, which causes an additional TypeError.
        # This needs to be fixes on pydantic's end, I think.
        instantiate(WithVal)

    assert instantiate(WithVal(10)) == 10


def test_basic_pydantic_validation_on_class():
    NoVal = builds(NeedsPosInt, -100)
    WithVal = builds(NeedsPosInt, x=-100, zen_wrappers=validates_with_pydantic)
    assert instantiate(NoVal).x == -100

    with pytest.raises((ValidationError, TypeError)):
        # Unfortunately the interface for `pydantic.ValidationError` is weird..
        # and breaks Hydra's error handling, which causes an additional TypeError.
        # This needs to be fixes on pydantic's end, I think.
        instantiate(WithVal)

    WithVal = builds(NeedsPosInt, x=100, zen_wrappers=validates_with_pydantic)
    out = instantiate(WithVal)
    assert isinstance(out, NeedsPosInt)
    assert out.x == 100


class F:
    pass


class G:
    pass


def tuple_of_classes(x: Tuple[F, G]):
    return x


def test_pydantic_with_arbitrary_allowed_types():
    instantiate(builds(tuple_of_classes, [builds(F), builds(F)]))  # should pass

    with pytest.raises((ValidationError, TypeError)):
        instantiate(
            builds(
                tuple_of_classes,
                [builds(F), builds(F)],
                zen_wrappers=validates_with_pydantic,
                hydra_convert="all",
            )
        )

    f, g = instantiate(
        builds(
            tuple_of_classes,
            [builds(F), builds(G)],
            zen_wrappers=validates_with_pydantic,
            hydra_convert="all",
        )
    )
    assert isinstance(f, F)
    assert isinstance(g, G)
