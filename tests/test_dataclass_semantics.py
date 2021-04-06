from dataclasses import is_dataclass

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_utils import builds, instantiate


def f_three_vars(x, y, z):
    return x, y, z


def f(x, y):
    return x, y


@given(full_sig=st.booleans(), partial=st.booleans())
def test_builds_produces_dataclass(full_sig: bool, partial: bool):

    if full_sig:
        Builds_f = builds(f, populate_full_signature=full_sig, hydra_partial=partial)
    else:
        Builds_f = builds(f, x=None, y=None, hydra_partial=partial)
    assert is_dataclass(Builds_f)
    out = Builds_f(x=1.0, y=-1.0)
    assert out.x == 1.0
    assert out.y == -1.0


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
