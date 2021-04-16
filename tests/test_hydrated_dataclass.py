# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from dataclasses import is_dataclass

import pytest

from hydra_zen import hydrated_dataclass, instantiate


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
