# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, to_yaml
from tests import valid_hydra_literals


def x_is_pos_only(x, /):
    return x


def xy_are_pos_only(x, y, /):
    return x, y


@pytest.mark.parametrize("func", [x_is_pos_only])
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_runtime_validation_pos_only_not_nameable(func, full_sig, partial):
    with pytest.raises(TypeError):
        builds(func, x=10, zen_partial=partial, populate_full_signature=full_sig)


@given(
    x=valid_hydra_literals,
    full_sig=st.booleans(),
    partial=st.booleans(),
)
def test_roundtrip_pos_only(x, full_sig: bool, partial: bool):
    cfg = builds(
        x_is_pos_only, x, populate_full_signature=full_sig, zen_partial=partial
    )

    out = instantiate(cfg)

    assert len(inspect.signature(cfg).parameters) == 0

    if partial:
        out = out()
    assert out == x

    out = instantiate(OmegaConf.create(to_yaml(cfg)))
    if partial:
        out = out()
    assert out == x


@given(full_sig=st.booleans())
def test_pos_only_with_partial_is_not_required(full_sig):
    cfg = builds(x_is_pos_only, populate_full_signature=full_sig, zen_partial=True)
    assert instantiate(cfg)(1) == 1
    assert len(inspect.signature(cfg).parameters) == 0

    cfg = builds(xy_are_pos_only, 1, populate_full_signature=full_sig, zen_partial=True)
    assert instantiate(cfg)(2) == (1, 2)
    assert len(inspect.signature(cfg).parameters) == 0
