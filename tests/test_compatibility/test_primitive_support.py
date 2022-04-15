# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


from dataclasses import dataclass

import pytest
from omegaconf import OmegaConf, ValidationError

from hydra_zen import builds, instantiate
from hydra_zen._compatibility import HYDRA_SUPPORTS_BYTES


@dataclass
class C:
    x: bytes = b"123"


def test_hydra_supports_bytes():

    if HYDRA_SUPPORTS_BYTES:
        OmegaConf.create(C)
    else:
        with pytest.raises(ValidationError):
            OmegaConf.create(C)

    assert instantiate(builds(C, populate_full_signature=True)).x == C.x
