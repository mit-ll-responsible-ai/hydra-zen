# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT


from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from hydra_zen import builds, instantiate


@dataclass
class C_bytes:
    x: bytes = b"123"


@dataclass
class C_paths:
    x: Path = Path.cwd()


def test_hydra_supports_bytes():
    OmegaConf.create(C_bytes)  # type: ignore
    assert instantiate(builds(C_bytes, populate_full_signature=True)).x == C_bytes.x


def test_hydra_supports_paths():
    out = OmegaConf.create(C_paths)  # type: ignore
    assert isinstance(out.x, Path)

    builds_out = instantiate(builds(C_paths, populate_full_signature=True))
    assert builds_out.x == C_paths.x
