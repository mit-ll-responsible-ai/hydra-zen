# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import pytest
from beartype.cave import RegexTypes  # type: ignore
from beartype.vale import Is  # type: ignore
from typing_extensions import Annotated

from hydra_zen.third_party.beartype import validates_with_beartype


@pytest.mark.parametrize(
    "custom_type, good_val, bad_val",
    [
        (RegexTypes, "abc+", 22),
        (Annotated[str, Is[lambda text: 2 == len(text)]], "hi", "bye"),
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
