# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any

import pytest
from omegaconf import OmegaConf

from hydra_zen import DefaultBuilds


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (list[int], list[int]),
        (tuple[str, str], tuple[str, str]),
        (dict[str, int], dict[str, int]),
        (set[str], Any),
    ],
)
def test_sanitized_type_expected_behavior(in_type, expected_type):
    # tests collections-as-generics introduced in py39
    actual = DefaultBuilds._sanitized_type(in_type)
    assert actual is expected_type or actual == expected_type

    @dataclass
    class Tmp:
        x: expected_type

    OmegaConf.structured(Tmp)
