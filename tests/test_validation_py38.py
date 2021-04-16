# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds


def x_is_pos_only(x, /):
    pass


@pytest.mark.parametrize("func", [x_is_pos_only])
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_user_specified_arg_is_not_in_sig(func, full_sig, partial):
    with pytest.raises(TypeError):
        builds(func, x=10, hydra_partial=partial, populate_full_signature=full_sig)
