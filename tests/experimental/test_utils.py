from collections import deque
from typing import Deque, List, Sequence, Tuple

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, instantiate
from hydra_zen.experimental.utils import convert_sequences


def f(x):
    return x


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),
        (str, str),
        (Tuple[int, int], tuple),
        (tuple, tuple),
        (Deque[str], deque),
        (List[str], list),
        (Sequence[str], list),
    ],
)
def test_convert_sequences_against_many_types(in_type, expected_type):
    f.__annotations__["x"] = in_type

    @given(x=st.from_type(in_type), as_named=st.booleans())
    def run(x, as_named: bool):
        args = ()
        kwargs = {}
        if as_named:
            kwargs["x"] = x
        else:
            args = (x,)
        out = instantiate(
            builds(
                f, *args, **kwargs, zen_wrappers=convert_sequences, hydra_convert="all"
            )
        )
        assert isinstance(out, expected_type)
        assert out == expected_type(x)

    run()
