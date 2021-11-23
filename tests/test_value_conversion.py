# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from collections import Counter, deque
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, List, Set, Union

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, assume, given, settings
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, make_config, to_yaml
from hydra_zen.structured_configs._value_conversion import (
    ZEN_SUPPORTED_PRIMITIVES,
    ZEN_VALUE_CONVERSION,
)


def test_supported_primitives_in_sync_with_value_conversion():
    assert len(set(ZEN_SUPPORTED_PRIMITIVES)) == len(ZEN_SUPPORTED_PRIMITIVES)
    assert set(ZEN_SUPPORTED_PRIMITIVES) == set(ZEN_VALUE_CONVERSION)


class Shake(Enum):
    VANILLA = 7
    CHOCOLATE = 4
    COOKIES = 9
    MINT = 3


@pytest.mark.parametrize(
    "zen_supported_type",
    (
        # Hydra supported primitives
        int,
        float,
        bool,
        str,
        type(None),
        Shake,
        List[Union[int, List[int], Dict[int, int]]],
        Dict[int, Union[int, List[int], Dict[int, int]]],
        # hydra-zen supported primitives
        set,
        frozenset,
        FrozenSet[Union[int, complex]],
        Set[Union[int, str, complex, Path]],
        complex,
        Path,
        bytes,
        bytearray,
        deque,
        range,
        Counter,
    ),
)
@settings(
    deadline=None,
    max_examples=40,
    suppress_health_check=(HealthCheck.data_too_large, HealthCheck.too_slow),
)
@given(data=st.data(), as_builds=st.booleans(), via_yaml=st.booleans())
def test_value_supported_via_config_maker_functions(
    zen_supported_type, data: st.DataObject, as_builds: bool, via_yaml: bool
):
    value = data.draw(st.from_type(zen_supported_type))
    if isinstance(value, str):
        assume(value.isascii())

    Conf = (
        make_config(a=value, hydra_convert="all")
        if not as_builds
        else builds(dict, a=value, hydra_convert="all")
    )

    if via_yaml:
        Conf = OmegaConf.structured(to_yaml(Conf))

    conf = instantiate(Conf)
    assert isinstance(conf["a"], type(value))
    if value == value:  # avoid nans
        if (
            not isinstance(value, range)
            and hasattr(value, "__iter__")
            and any(v != v for v in value)
        ):  # avoid nans in collections
            pass
        else:
            assert conf["a"] == value
