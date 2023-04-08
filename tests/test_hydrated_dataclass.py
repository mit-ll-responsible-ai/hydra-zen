# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import is_dataclass

import pytest

from hydra_zen import hydrated_dataclass, instantiate
from hydra_zen.structured_configs._globals import (
    PARTIAL_FIELD_NAME,
    ZEN_PARTIAL_FIELD_NAME,
)


def f1(x, y):
    pass


def f2(x, y, z):
    return x, y, z


@pytest.mark.parametrize("conf2_sig", [True, False])
@pytest.mark.parametrize("conf3_sig", [True, False])
def test_chained_inheritance(conf2_sig, conf3_sig):
    @hydrated_dataclass(f1)
    class Conf1:
        x: int = 1

    @hydrated_dataclass(f1, populate_full_signature=conf2_sig)
    class Conf2(Conf1):
        y: int = 2

    @hydrated_dataclass(f2, populate_full_signature=True)
    class Conf3(Conf2):
        z: int = 3

    assert is_dataclass(Conf1)
    assert is_dataclass(Conf2)
    assert is_dataclass(Conf3)

    assert issubclass(Conf2, Conf1)
    assert issubclass(Conf3, Conf1)
    assert issubclass(Conf3, Conf2)
    assert instantiate(Conf3) == (1, 2, 3)


def test_pos_args():
    @hydrated_dataclass(f2, 1, 2)
    class Conf:
        z: int = 3

    assert instantiate(Conf) == (1, 2, 3)


@pytest.mark.parametrize("zen_partial", [None, False, True])
def test_partial(zen_partial):
    kw = {}
    if zen_partial is not None:
        kw["zen_partial"] = zen_partial

    @hydrated_dataclass(f2, 1, 2, **kw)
    class Conf:
        z: int = 3

    if zen_partial is None:
        assert not hasattr(Conf, PARTIAL_FIELD_NAME) and not hasattr(
            Conf, ZEN_PARTIAL_FIELD_NAME
        )
    else:
        assert getattr(Conf, PARTIAL_FIELD_NAME) is zen_partial
