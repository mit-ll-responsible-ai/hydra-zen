# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import string
from pathlib import Path

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from omegaconf import MISSING as omega_MISSING, OmegaConf

from hydra_zen import (
    MISSING as zen_MISSING,
    builds,
    instantiate,
    just,
    load_from_yaml,
    save_as_yaml,
    to_yaml,
)


@given(resolve=st.none() | st.booleans(), sort_keys=st.none() | st.booleans())
def test_to_yaml_matches_omegaconf(resolve, sort_keys):
    kwargs = {}
    if resolve is not None:
        kwargs["resolve"] = resolve

    if sort_keys is not None:
        kwargs["sort_keys"] = sort_keys

    actual = to_yaml(builds(dict, a="1", b="${a}"), **kwargs)
    expected = OmegaConf.to_yaml(builds(dict, a="1", b="${a}"), **kwargs)
    assert actual == expected


def test_MISSING_is_alias_from_omegaconf():
    assert omega_MISSING is zen_MISSING


@settings(max_examples=10, deadline=None)
@given(fname=st.text(string.ascii_letters, min_size=1), value=st.integers(-2, 2))
@pytest.mark.usefixtures("cleandir")
def test_yaml_io_roundtrip(fname, value):
    conf = builds(dict, hello=10, goodbye=value)
    save_as_yaml(conf, fname)
    loaded_conf = load_from_yaml(fname)

    assert instantiate(conf) == instantiate(loaded_conf)


def test_to_yaml_applies_just():
    data = [1 + 2j, Path.cwd(), "hi"]
    assert to_yaml(just(data)) == to_yaml(data)


def test_save_yaml_applies_just(tmp_path):
    data = {"a": [1 + 2j, "hi", 3j]}
    save_as_yaml(data, tmp_path / "cfg.yml")
    out = dict(instantiate(load_from_yaml(tmp_path / "cfg.yml")))
    assert out == data
