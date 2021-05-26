# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import hypothesis.strategies as st
from hypothesis import given
from omegaconf import MISSING as omega_MISSING, OmegaConf

from hydra_zen import MISSING as zen_MISSING, builds, to_yaml


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
