# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, make_config, to_yaml, zen
from hydra_zen.errors import HydraZenValidationError
from tests import valid_hydra_literals


def x_is_pos_only(x, /):
    return x


def xy_are_pos_only(x, y, /):
    return x, y


def test_zen_decorator_with_positional_only_args():
    zen_f = zen(x_is_pos_only)
    with pytest.raises(
        HydraZenValidationError,
        match=r"has 1 positional-only arguments, but `cfg` specifies 0",
    ):
        zen_f.validate(make_config())

    cfg = builds(x_is_pos_only, 1)
    zen_f.validate(cfg)
    assert zen_f(cfg) == 1


@pytest.mark.parametrize("func", [x_is_pos_only])
@given(partial=st.none() | st.booleans(), full_sig=st.booleans())
def test_builds_runtime_validation_pos_only_not_nameable(func, full_sig, partial):
    with pytest.raises(TypeError):
        builds(func, x=10, zen_partial=partial, populate_full_signature=full_sig)


@given(
    x=valid_hydra_literals,
    full_sig=st.booleans(),
    partial=st.none() | st.booleans(),
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
