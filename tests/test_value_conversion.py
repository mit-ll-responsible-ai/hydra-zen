# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from pathlib import Path
from typing import Set, Union

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from hydra_zen import instantiate, make_config, to_yaml
from hydra_zen.structured_configs._value_conversion import (
    ZEN_SUPPORTED_PRIMITIVES,
    ZEN_VALUE_CONVERSION,
)


def test_supported_primitives_in_sync_with_value_conversion():
    assert len(set(ZEN_SUPPORTED_PRIMITIVES)) == len(ZEN_SUPPORTED_PRIMITIVES)
    assert set(ZEN_SUPPORTED_PRIMITIVES) == set(ZEN_VALUE_CONVERSION)


@pytest.mark.parametrize(
    "zen_supported_type", (set, Set[Union[int, str, complex, Path]], complex, Path)
)
@settings(deadline=None, max_examples=20)
@given(st.data())
def test_value_conversion(zen_supported_type, data: st.DataObject):
    value = data.draw(st.from_type(zen_supported_type))
    Conf = make_config(a=value)
    to_yaml(Conf)
    conf = instantiate(Conf)
    assert isinstance(conf.a, type(value))
    if value == value:  # avoid nans
        if hasattr(value, "__iter__") and any(
            v != v for v in value
        ):  # avoid nans in collections
            pass
        else:
            assert conf.a == value
