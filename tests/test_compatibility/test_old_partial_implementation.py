# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from typing import Any

from omegaconf import OmegaConf

from hydra_zen import get_target, instantiate, just, to_yaml


def f(*args, **kwargs):
    return args, kwargs


# These tests ensure that the implementation of zen_partial=True
# for hydra_zen < 0.3.0 still works
@dataclass
class OldPartial:
    _target_: str = "hydra_zen.funcs.partial"
    _partial_target_: Any = just(f)
    _args_: Any = ("pos",)
    a: int = 1


def test_old_partial_instantiates():
    args, kwargs = instantiate(OldPartial)(b=2)
    assert args == ("pos",)
    assert kwargs == {"a": 1, "b": 2}


def test_old_partial_from_yaml():
    yaml = to_yaml(OldPartial)
    args, kwargs = instantiate(OmegaConf.create(yaml))(b=2)
    assert args == ("pos",)
    assert kwargs == {"a": 1, "b": 2}


def test_old_partial_get_target():
    assert get_target(OldPartial) is f
