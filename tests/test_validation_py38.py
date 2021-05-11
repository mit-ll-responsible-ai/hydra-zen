# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, instantiate, to_yaml
from tests import valid_hydra_literals


def x_is_pos_only(x, /):
    return x


@pytest.mark.parametrize("func", [x_is_pos_only])
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_user_specified_arg_is_not_in_sig(func, full_sig, partial):
    with pytest.raises(TypeError):
        builds(func, x=10, hydra_partial=partial, populate_full_signature=full_sig)


@given(
    x=valid_hydra_literals,
    full_sig=st.booleans(),
    partial=st.booleans(),
)
def test_roundtrip_pos_only(x, full_sig: bool, partial: bool):
    cfg = builds(
        x_is_pos_only, x, populate_full_signature=full_sig, hydra_partial=partial
    )
    to_yaml(cfg)  # shouldn't crash
    out = instantiate(cfg)

    if partial:
        out = out()
    assert out == x


@given(full_sig=st.booleans(), partial=st.booleans())
def test_pos_only_sig_parsing(full_sig: bool, partial: bool):
    cfg = builds(x_is_pos_only, populate_full_signature=full_sig, hydra_partial=partial)
    assert len(inspect.signature(cfg).parameters) == 0
