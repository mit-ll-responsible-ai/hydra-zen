# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect

import hypothesis.strategies as st
import jax
import jax.numpy as jnp
import pytest
from hypothesis import assume, given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, just, to_yaml


def test_builds_roundtrip_with_ufunc():
    assert instantiate(builds(jnp.add, zen_partial=True))(1.0, 2.0) == jnp.array(3.0)


jax_objects = [
    jax.jit,
    jax.jacfwd,
    jax.jacrev,
    jax.grad,
    jax.vmap,
    jax.pmap,
    jnp.add,
    jnp.matmul,
    jnp.exp,
    jnp.linalg.norm,
    jnp.array,
]


@pytest.mark.parametrize("obj", jax_objects)
def test_just_roundtrip(obj):
    assert instantiate(just(obj)) is obj


@pytest.mark.parametrize("target", jax_objects)
@given(partial=st.booleans(), full_sig=st.booleans())
def test_fuzz_build_validation_against_a_bunch_of_common_objects(
    target, partial: bool, full_sig: bool
):
    doesnt_have_sig = False
    try:
        inspect.signature(target)
    except ValueError:
        doesnt_have_sig = True

    if doesnt_have_sig and full_sig:
        assume(False)
    conf = builds(target, zen_partial=partial, populate_full_signature=full_sig)

    OmegaConf.create(to_yaml(conf))  # ensure serializable

    if partial:
        instantiate(conf)  # ensure instantiable
