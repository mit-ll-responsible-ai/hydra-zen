# Copyright (c) 2021 Massachusetts Institute of Technology

import numpy as np
import pytest

from hydra_zen import builds, instantiate, just
from hydra_zen.structured_configs._utils import safe_name


def test_builds_roundtrip_with_ufunc():
    assert instantiate(builds(np.add, hydra_partial=True))(1.0, 2.0) == np.array(3.0)


@pytest.mark.parametrize(
    "obj",
    [
        np.array,
        np.add,
        np.ufunc,  # ufuncs work!
        np.linalg.norm,
    ],
)
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
