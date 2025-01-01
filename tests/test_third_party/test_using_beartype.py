# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from beartype.cave import RegexTypes
from beartype.vale import Is
from typing_extensions import Annotated

from hydra_zen.third_party.beartype import validates_with_beartype


def func(x: int) -> float: ...


@pytest.mark.parametrize(
    "custom_type, good_val, bad_val",
    [
        (RegexTypes, "abc+", 22),
        (Annotated[str, Is[lambda text: 2 == len(text)]], "hi", "bye"),
        # (Partial[float], partial(func), func),
        # (Builds[Type[float]], builds(func), func),
    ],
)
def test_beartype_specific_fields(custom_type, good_val, bad_val):
    def f(x):
        pass

    f.__annotations__["x"] = custom_type
    bear_hugged_f = validates_with_beartype(f)

    bear_hugged_f(good_val)  # ok
    with pytest.raises(Exception):
        bear_hugged_f(bad_val)

    class A:
        def __init__(self, x) -> None:
            pass

    A.__init__.__annotations__["x"] = custom_type
    validates_with_beartype(A)

    A(good_val)
    with pytest.raises(Exception):
        A(bad_val)
