# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

from copy import deepcopy
from dataclasses import FrozenInstanceError, is_dataclass

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
        Builds_f = builds(f, populate_full_signature=full_sig, zen_partial=partial)
    else:
        Builds_f = builds(
            f, x=None, y=None, zen_partial=partial, populate_full_signature=full_sig
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
        zen_partial=partial,
        populate_full_signature=full_sig,
        builds_bases=(conf_2,),
    )

    # checks subclass relationships
    assert conf_3.__mro__[:3] == (conf_3, conf_2, conf_1)

    out = instantiate(conf_3)
    if partial:
        out = out()  # resolve partial

    assert out == (1, 2, 3)


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_pos_args_with_inheritance(full_sig, partial):

    conf_1 = builds(f_three_vars, 1, 2)
    conf_2 = builds(
        f_three_vars,
        z=3,
        zen_partial=partial,
        populate_full_signature=full_sig,
        builds_bases=(conf_1,),
    )

    # checks subclass relationships
    assert conf_2.__mro__[:2] == (conf_2, conf_1)

    out = instantiate(conf_2)
    if partial:
        out = out()  # resolve partial

    assert out == (1, 2, 3)


def f_3(x):
    pass


def test_frozen_via_builds():

    conf_f = builds(f, x=2, frozen=True)()

    with pytest.raises(FrozenInstanceError):
        conf_f.x = 3


def test_frozen_via_hydrated_dataclass():
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
def test_mutable_defaults_generated_from_factory(mutable):
    Conf = builds(dict, x=mutable)
    mutable = deepcopy(mutable)

    instance1 = Conf()
    if isinstance(mutable, dict):
        instance1.x.pop(0)
    else:
        instance1.x.pop()

    # mutation via instance1 should not affect other instances of `Conf`
    instance2 = Conf()
    assert instance2.x == mutable

    # make sure hydra behavior is appropriate
    out_Conf = instantiate(Conf)["x"]
    assert out_Conf == mutable

    out_inst = instantiate(instance2)["x"]
    assert out_inst == mutable


class NotSet:
    pass


@pytest.mark.parametrize("recursive", [True, False, NotSet])
@pytest.mark.parametrize("convert", ["none", "partial", "all", NotSet])
@pytest.mark.parametrize("via_builds", [True, False])
def test_hydra_settings_can_be_inherited(recursive, convert, via_builds):
    kwargs = {}
    if recursive is not NotSet:
        kwargs["hydra_recursive"] = recursive

    if convert is not NotSet:
        kwargs["hydra_convert"] = convert

    if via_builds:
        Base = builds(dict, **kwargs)
        Child = builds(dict, builds_bases=(Base,))
    else:

        @hydrated_dataclass(target=dict, **kwargs)
        class Base:
            pass

        @hydrated_dataclass(target=dict)
        class Child(Base):
            pass

    if recursive is not NotSet:
        assert Child._recursive_ is Base._recursive_
    else:
        assert not hasattr(Child, "_recursive_")

    if convert is not NotSet:
        assert Child._convert_ is Base._convert_
    else:
        assert not hasattr(Child, "_convert_")
