# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from hydra_zen import builds, instantiate, just, make_custom_builds_fn
from hydra_zen.funcs import get_obj, zen_processing
from hydra_zen.structured_configs._type_guards import (
    is_builds,
    is_just,
    is_partial_builds,
)
from hydra_zen.typing import Builds, Just, Partial, ZenPartialBuilds
from hydra_zen.typing._implementations import HydraPartialBuilds


@pytest.mark.parametrize(
    "fn, protocol",
    [
        (just, Just),
        (builds, Builds),
        (partial(builds, zen_partial=True), (ZenPartialBuilds, HydraPartialBuilds)),
        (partial(builds, zen_partial=True, zen_meta=dict(y=1)), ZenPartialBuilds),
    ],
)
def test_runtime_checkability_of_protocols(fn, protocol):
    Conf = fn(dict)
    assert isinstance(Conf, protocol)

    conf = Conf() if fn is not just else Conf
    assert isinstance(conf, protocol)


def test_Builds_is_not_ZenPartialBuilds():
    Conf = builds(dict)
    assert not isinstance(Conf, ZenPartialBuilds)

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
        (
            make_custom_builds_fn(zen_partial=True, zen_meta=dict(_y=None)),
            ZenPartialBuilds,
        ),
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
class AZenPartial:
    _target_: Any = zen_processing
    _zen_target: Any = "builtins.int"
    _zen_partial: bool = True


@dataclass
class NotJust:
    _target_: Any = "builtins.dict"  # wrong target
    path: Any = "builtins.int"


@dataclass
class NotZenPartial:
    _target_: Any = "builtins.dict"  # wrong target
    _zen_target: Any = "builtins.int"
    _zen_partial: bool = True


@dataclass
class NotZenPartial2:
    _target_: Any = "builtins.dict"
    # MISSING _zen_target
    _zen_partial: bool = True


@dataclass
class NotZenPartial3:
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
        (builds(int, zen_partial=True, zen_meta=dict(a=1)), True, False, True),
        (AZenPartial, True, False, True),
        (NotJust, True, False, False),
        (NotZenPartial, True, False, False),
        (NotZenPartial2, True, False, False),
        (NotZenPartial3, True, False, False),
    ],
)
def test_protocol_checkers(x, yes_builds, yes_just, yes_partial):
    assert is_builds(x) is yes_builds
    assert is_just(x) is yes_just
    assert is_partial_builds(x) is yes_partial

    if yes_builds or yes_just or yes_partial:
        instantiate(x)


def test_partial_protocol():
    assert isinstance(partial(int), Partial)
    assert not isinstance(print, Partial)


def test_parameterized_partial_regression():
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/352
    assert Partial[int].__origin__ is Partial  # type: ignore
