# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from copy import deepcopy
from dataclasses import is_dataclass

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, hydrated_dataclass, instantiate


def f_three_vars(x, y, z):
    return x, y, z


def f(x, y, z: int = 3):
    return x, y, z


@given(full_sig=st.booleans(), partial=st.booleans())
def test_builds_produces_dataclass(full_sig: bool, partial: bool):

    if full_sig and not partial:
        Builds_f = builds(f, populate_full_signature=full_sig, hydra_partial=partial)
    else:
        Builds_f = builds(
            f, x=None, y=None, hydra_partial=partial, populate_full_signature=full_sig
        )
    assert is_dataclass(Builds_f)
    out = Builds_f(x=1.0, y=-1.0)
    assert out.x == 1.0
    assert out.y == -1.0

    if full_sig:
        assert out.z == 3


def f_2(x, y, z):
    pass


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_chain_builds_of_targets_with_common_interfaces(full_sig, partial):

    # Note that conf_1 and conf_2 target `f` whereas conf_3 targets `f_three_vars`,
    # which have identical interfaces.
    conf_1 = builds(f_2, x=1)
    conf_2 = builds(f_2, y=2, builds_bases=(conf_1,))
    conf_3 = builds(
        f_three_vars,
        z=3,
        hydra_partial=partial,
        populate_full_signature=full_sig,
        builds_bases=(conf_2,),
    )

    # checks subclass relationships
    assert conf_3.__mro__[:3] == (conf_3, conf_2, conf_1)

    out = instantiate(conf_3)
    if partial:
        out = out()  # resolve partial

    assert out == (1, 2, 3)


def f_3(x):
    pass


def test_frozen():
    from dataclasses import FrozenInstanceError

    conf_f = builds(f, x=2, frozen=True)()

    with pytest.raises(FrozenInstanceError):
        conf_f.x = 3

    @hydrated_dataclass(f, frozen=True)
    class Conf_f:
        x: int = 2

    conf_f = Conf_f()

    with pytest.raises(FrozenInstanceError):
        conf_f.x = 3


@given(
    mutable=st.lists(st.integers(), min_size=1)
    | st.dictionaries(st.integers(), st.integers(), min_size=1).map(
        lambda x: {0: 0, **x}
    )
    | st.sets(st.integers(), min_size=1)
)
def test_mutable_values(mutable):
    Conf = builds(dict, x=mutable)
    mutable = deepcopy(mutable)

    instance1 = Conf()
    if isinstance(mutable, dict):
        instance1.x.pop(0)
    else:
        instance1.x.pop()

    instance2 = Conf()
    assert instance2.x == mutable
