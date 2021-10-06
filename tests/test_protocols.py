# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from hydra_zen import builds, instantiate, just
from hydra_zen.funcs import get_obj, zen_processing
from hydra_zen.structured_configs._implementations import (
    is_builds,
    is_just,
    is_partial_builds,
)
from hydra_zen.typing import Builds, Just, PartialBuilds


@pytest.mark.parametrize(
    "fn, protocol",
    [
        (just, Just),
        (builds, Builds),
        (partial(builds, zen_partial=True), PartialBuilds),
    ],
)
def test_runtime_checkability_of_protocols(fn, protocol):
    Conf = fn(dict)
    assert isinstance(Conf, protocol)

    conf = Conf()
    assert isinstance(conf, protocol)


def test_Builds_is_not_PartialBuilds():
    Conf = builds(dict)
    assert not isinstance(Conf, PartialBuilds)

    PConf = builds(dict, zen_partial=True)
    assert isinstance(PConf, Builds)


def test_targeted_dataclass_is_Builds():
    @dataclass
    class NonTargeted:
        pass

    @dataclass
    class Targeted:
        _target_: str = "hello"

    assert not isinstance(NonTargeted, Builds)
    assert not isinstance(NonTargeted(), Builds)
    assert isinstance(Targeted, Builds)
    assert isinstance(Targeted(), Builds)


@pytest.mark.parametrize(
    "fn,protocol",
    [
        (just, Just),
        (partial(builds, zen_partial=True), PartialBuilds),
    ],
)
def test_protocol_target_is_correct(fn, protocol):
    assert fn(int)._target_ == protocol._target_


@dataclass
class ABuilds:
    _target_: Any = int


@dataclass
class AJust:
    _target_: Any = get_obj
    path: Any = "builtins.int"


@dataclass
class APartial:
    _target_: Any = zen_processing
    _zen_target: Any = "builtins.int"
    _zen_partial: bool = True


@dataclass
class NotJust:
    _target_: Any = "builtins.dict"  # wrong target
    path: Any = "builtins.int"


@dataclass
class NotPartial:
    _target_: Any = "builtins.dict"  # wrong target
    _zen_target: Any = "builtins.int"
    _zen_partial: bool = True


@dataclass
class NotPartial2:
    _target_: Any = "builtins.dict"
    # MISSING _zen_target
    _zen_partial: bool = True


@dataclass
class NotPartial3:
    # partial is False
    _target_: Any = zen_processing
    _zen_target: Any = "builtins.int"
    _zen_partial: bool = False  # <- False!


@pytest.mark.parametrize(
    "x,yes_builds,yes_just,yes_partial",
    [
        (1, False, False, False),
        (builds(int), True, False, False),
        (just(int), True, True, False),
        (AJust, True, True, False),
        (builds(int, zen_partial=True), True, False, True),
        (APartial, True, False, True),
        (NotJust, True, False, False),
        (NotPartial, True, False, False),
        (NotPartial2, True, False, False),
        (NotPartial3, True, False, False),
    ],
)
def test_protocol_checkers(x, yes_builds, yes_just, yes_partial):
    assert is_builds(x) is yes_builds
    assert is_just(x) is yes_just
    assert is_partial_builds(x) is yes_partial

    if yes_builds or yes_just or yes_partial:
        instantiate(x)
