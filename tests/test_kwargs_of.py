# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from inspect import signature

import pytest

from hydra_zen import BuildsFn, instantiate, kwargs_of


def test_basic():
    Conf = kwargs_of(lambda x, y: None)
    assert set(signature(Conf).parameters) == {"x", "y"}
    out = instantiate(Conf(x=-9, y=10))
    assert isinstance(out, dict)
    assert out == dict(x=-9, y=10)


@pytest.mark.parametrize(
    "exclude,params",
    [
        ([0], {"y"}),
        (["x", -1], set()),
    ],
)
def test_exclude(exclude, params):
    Conf = kwargs_of((lambda x, y: None), zen_exclude=exclude)
    assert set(signature(Conf).parameters) == params


def test_dataclass_options():
    Conf = kwargs_of((lambda x, y: None), zen_dataclass={"cls_name": "foo"})
    assert Conf.__name__ == "foo"


def test_dataclass_options_via_cls_defaults():
    class Moo(BuildsFn):
        _default_dataclass_options_for_kwargs_of = {"cls_name": "bar"}

    Conf1 = kwargs_of((lambda: None), zen_dataclass={"cls_name": "foo"})
    assert Conf1.__name__ == "foo"

    Conf2 = Moo.kwargs_of((lambda: None))
    assert Conf2.__name__ == "bar"
