# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given

from hydra_zen import builds, instantiate, just
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
    builds(target, hydra_partial=partial, populate_full_signature=full_sig)
