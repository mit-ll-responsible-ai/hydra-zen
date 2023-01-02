# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
import random
from functools import partial

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given
from omegaconf import OmegaConf

from hydra_zen import builds, get_target, instantiate, just, to_yaml
from hydra_zen.structured_configs._utils import safe_name


def test_builds_roundtrip_with_ufunc():
    assert instantiate(builds(np.add, zen_partial=True))(1.0, 2.0) == np.array(3.0)


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
    "hydra_zen_func", [builds, partial(builds, zen_partial=True), just]
)
def test_get_target_roundtrip(obj, hydra_zen_func):
    conf = hydra_zen_func(obj)
    assert get_target(conf) is obj
    assert get_target(OmegaConf.create(to_yaml(conf))) is obj


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
    conf = builds(target, zen_partial=partial, populate_full_signature=full_sig)

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


def test_obfuscated_modules_dont_get_mixed_up():
    # Both `numpy.random.random` and `random.random` have missing
    # __module__ attributes, and we have to use specialized logic
    # to fetch these.
    #
    # This test ensures that we never misattribute functions with
    # the same name, to the wrong module
    assert builds(np.random.random)._target_.startswith("numpy")
    assert builds(random.random)._target_.startswith("random")

    assert get_target(builds(np.random.random)) is np.random.random
    assert get_target(builds(random.random)) is random.random


SHOULD_NOT_IMPORT_NUMPY = """
import sys

def test_no_numpy_import():
    assert "numpy" not in sys.modules

    from hydra_zen import builds, launch, make_config
    from hydra_zen.funcs import get_obj

    make_config(a=1)
    builds(dict, a=make_config)  # specifically exercises auto-just

    # jogs imports through potential obfuscated modules
    try:
        get_obj(path="blahblah")
    except ImportError:
        pass

    assert "numpy" not in sys.modules
"""


def test_hydra_zen_is_not_the_first_to_import_numpy(pytester):
    # We only import numpy if the user did so first.
    pytester.makepyfile(SHOULD_NOT_IMPORT_NUMPY)
    pytester.makeconftest("")
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1, failed=0)
