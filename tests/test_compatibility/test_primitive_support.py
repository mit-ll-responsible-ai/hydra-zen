# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from pathlib import Path

import pytest
from omegaconf import OmegaConf, ValidationError

from hydra_zen import builds, instantiate
from hydra_zen._compatibility import HYDRA_SUPPORTS_BYTES, HYDRA_SUPPORTS_Path


@dataclass
class C_bytes:
    x: bytes = b"123"


@dataclass
class C_paths:
    x: Path = Path.cwd()


def test_hydra_supports_bytes():

    if HYDRA_SUPPORTS_BYTES:
        OmegaConf.create(C_bytes)  # type: ignore
    else:
        with pytest.raises(ValidationError):
            OmegaConf.create(C_bytes)  # type: ignore

    assert instantiate(builds(C_bytes, populate_full_signature=True)).x == C_bytes.x


def test_hydra_supports_paths():

    if HYDRA_SUPPORTS_Path:
        out = OmegaConf.create(C_paths)  # type: ignore
        assert isinstance(out.x, Path)
    else:
        with pytest.raises(ValidationError):
            OmegaConf.create(C_paths)  # type: ignore

    builds_out = instantiate(builds(C_paths, populate_full_signature=True))
    assert builds_out.x == C_paths.x
