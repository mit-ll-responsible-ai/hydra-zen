# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import functools
from pathlib import Path

import pytest

from hydra_zen import instantiate, just, make_config
from tests import is_same


class A:
    @classmethod
    def class_method(cls):
        pass


def f(x: int):
    pass


@functools.lru_cache(maxsize=None)
def func_with_cache(x: int):
    pass


# this function is tested more rigorously via test_value_conversion
@pytest.mark.parametrize(
    "obj",
    [
        bytearray([1, 2, 3]),
        1 + 2j,
        Path.cwd(),
        A,
        f,
        func_with_cache,
        A.class_method,
        functools.partial(f, x=1),
    ],
)
def test_just_roundtrip(obj):
    out = instantiate(just(obj))

    if callable(out):
        assert is_same(out, obj)
    else:
        assert out == obj


def test_just_of_structured_config_is_identity():
    cfg = make_config(x=1)
    assert just(cfg) is cfg
