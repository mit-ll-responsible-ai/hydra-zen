# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from functools import partial
import inspect

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, just, to_yaml, get_target
from hydra_zen.structured_configs._utils import safe_name


def test_builds_roundtrip_with_ufunc():
    assert instantiate(builds(np.add, hydra_partial=True))(1.0, 2.0) == np.array(3.0)


numpy_objects = [
    np.array,
    np.dtype,
    np.add,
    np.ufunc,  # ufuncs work!
    np.linalg.norm,
    np.linalg.linalg.eigvalsh,
    np.reshape,
    np.random.rand,
    np.random.Generator,
    np.testing.assert_allclose,
    np.polynomial.Polynomial,
    np.polynomial.polynomial.polyadd,
]


@pytest.mark.parametrize("obj", numpy_objects)
@pytest.mark.parametrize(
    "hydra_zen_func", [builds, partial(builds, hydra_partial=True), just]
)
def test_get_target_roundtrip(obj, hydra_zen_func):
    assert get_target(hydra_zen_func(obj)) is obj


@pytest.mark.parametrize("obj", numpy_objects)
def test_just_roundtrip(obj):
    assert instantiate(just(obj)) is obj


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (np.add, "add"),
        (np.shape, "shape"),
        (np.array, "array"),
        (np.linalg.norm, "norm"),
    ],
)
def test_safename_known(obj, expected_name):
    assert safe_name(obj) == expected_name


@pytest.mark.parametrize("target", numpy_objects)
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
    conf = builds(target, hydra_partial=partial, populate_full_signature=full_sig)

    OmegaConf.create(to_yaml(conf))  # ensure serializable

    if partial:
        instantiate(conf)  # ensure instantiable


def f(reduction_fn=np.add):
    return reduction_fn


def test_ufunc_as_default_value():
    conf = builds(f, populate_full_signature=True)
    to_yaml(conf)  # check serializability
    assert instantiate(conf) is np.add


def test_ufunc_positional_args():
    assert instantiate(builds(np.add, 1.0, 2.0)) == 3.0
