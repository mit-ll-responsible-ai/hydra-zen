# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import random
from contextvars import ContextVar
from typing import Any, Dict, Optional

import pytest

from hydra_zen import zen

config: ContextVar[Optional[Dict[str, Any]]] = ContextVar("config", default=None)
var: ContextVar[Dict[str, Any]] = ContextVar("var", default=dict())


@pytest.fixture(autouse=True)
def clean_context_vars():
    assert config.get() is None
    assert var.get() == {}
    yield
    config.set(None)
    var.set({})


@pytest.mark.parametrize(
    "run_in_context",
    [
        True,
        pytest.param(False, marks=pytest.mark.xfail),
    ],
)
def test_context_isolation(run_in_context: bool):
    def foo(x: str, zen_cfg):
        config.set(zen_cfg)
        conf = var.get().copy()
        conf[str(random.randint(1, 100))] = random.randint(1, 100)
        var.set(conf)
        assert len(conf) == 1

    zfoo = zen(foo, run_in_context=run_in_context)

    for letter in "ab":
        zfoo(dict(x=letter))
        assert config.get() is None
        assert var.get() == dict()


async def async_func_run_in_context_not_supported():
    async def foo():
        ...

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
