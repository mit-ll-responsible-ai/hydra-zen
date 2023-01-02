# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pytest
from omegaconf import OmegaConf

from hydra_zen.structured_configs._utils import sanitized_type


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (list[int], List[int]),
        (tuple[str, str], Tuple[str, str]),
        (dict[str, int], Dict[str, int]),
        (set[str], Any),
    ],
)
def test_sanitized_type_expected_behavior(in_type, expected_type):
    # tests collections-as-generics introduced in py39
    actual = sanitized_type(in_type)
    assert actual is expected_type or actual == expected_type

    @dataclass
    class Tmp:
        x: expected_type

    OmegaConf.structured(Tmp)
