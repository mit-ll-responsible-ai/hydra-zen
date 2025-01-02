# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import random
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

import pytest

from hydra_zen import zen
from hydra_zen.errors import HydraZenValidationError

config: ContextVar[Optional[dict[str, Any]]] = ContextVar("config", default=None)
var: ContextVar[dict[str, Any]] = ContextVar("var", default=dict())


@pytest.fixture(autouse=True)
def clean_context_vars():
    assert config.get() is None
    assert var.get() == {}
    yield
    config.set(None)
    var.set({})


@dataclass
class TrackCall:
    num_calls: int = 0

    def __post_init__(self):
        self.funcs = []

    def __call__(self, fn) -> Any:
        self.num_calls += 1
        self.funcs.append(fn)
        return fn


@pytest.mark.parametrize(
    "run_in_context",
    [
        True,
        pytest.param(False, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize(
    "wrapper",
    [
        None,
        TrackCall,
    ],
)
def test_context_isolation(run_in_context: bool, wrapper: Optional[type[TrackCall]]):
    def foo(x: str, zen_cfg):
        config.set(zen_cfg)
        conf = var.get().copy()
        conf[str(random.randint(1, 100))] = random.randint(1, 100)
        var.set(conf)
        assert len(conf) == 1

    if wrapper is not None:
        wr = wrapper()
    else:
        wr = None

    zfoo = zen(foo, run_in_context=run_in_context, instantiation_wrapper=wr)

    for letter in "ab":
        zfoo(dict(x=letter))
        assert config.get() is None
        assert var.get() == dict()

    if isinstance(wr, TrackCall):
        assert wr.num_calls == 2


def test_async_func_run_in_context_not_supported():
    async def foo(): ...

    with pytest.raises(TypeError, match="not supported"):
        zen(foo, run_in_context=True)


@pytest.mark.parametrize(
    "run_in_context",
    [
        True,
        pytest.param(False, marks=pytest.mark.xfail),
    ],
)
def test_pre_call_shares_context_with_wrapped_func(run_in_context: bool):
    assert var.get() == {}

    def pre_call(cfg):
        var.set({"swagger": 22})

    def func():
        assert var.get() == {"swagger": 22}

    zen(func, pre_call=pre_call, run_in_context=run_in_context)({})
    assert var.get() == {}


def test_pre_call_run_in_its_own_context_is_forbidden():
    def f(x): ...

    with pytest.raises(HydraZenValidationError):
        zen(f, pre_call=zen(f, run_in_context=True), run_in_context=True)


def test_validation():
    with pytest.raises(TypeError, match="must be type"):
        zen(lambda x: x, run_in_context=None)  # type: ignore
