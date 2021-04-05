import string
from dataclasses import dataclass
from typing import Any, Dict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from raiden.framework.lightning import RaidenDataModule
from raiden.hydra_utils import instantiate
from torch.optim import Adam

from hydra_utils import builds, just
from tests import valid_hydra_literals

arbitrary_kwargs = st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=1),
    values=valid_hydra_literals,
)


def pass_through_kwargs(**kwargs):
    return kwargs


@given(kwargs=arbitrary_kwargs, full_sig=st.booleans())
def test_builds_roundtrip(kwargs, full_sig: bool):
    assert kwargs == instantiate(
        builds(pass_through_kwargs, **kwargs, populate_full_signature=full_sig)
    )


@given(
    partial_kwargs=arbitrary_kwargs,
    call_kwargs=arbitrary_kwargs,
    full_sig=st.booleans(),
)
def test_builds_roundtrip_with_partial(
    partial_kwargs: Dict[str, Any],
    call_kwargs: Dict[str, Any],
    full_sig: bool,
):
    partial_struct = instantiate(
        builds(
            pass_through_kwargs,
            hydra_partial=True,
            populate_full_signature=full_sig,
            **partial_kwargs,
        )
    )
    expected_kwargs = partial_kwargs.copy()
    expected_kwargs.update(call_kwargs)
    assert expected_kwargs == partial_struct(**call_kwargs)  # resolve partial


def test_builds_roundtrip_with_ufunc():
    assert instantiate(builds(np.add, hydra_partial=True))(1.0, 2.0) == np.array(3.0)


def test_documented_builds_simple_roundtrip_example():
    assert {"a": 1, "b": "x"} == instantiate(builds(dict, a=1, b="x"))


def test_documented_builds_roundtrip_partial_with_interpolation_example():
    import torch as tr
    from torch.optim import Adam

    @dataclass
    class ModuleConfig:
        learning_rate: float = 100.2
        optimizer: Any = builds(Adam, lr="${learning_rate}", hydra_partial=True)

    params = [tr.tensor(1.0)]
    config = instantiate(ModuleConfig)
    optim = config.optimizer(params)
    assert isinstance(optim, Adam)
    assert optim.defaults["lr"] == 100.2


def f(x, y=dict(a=2)):
    return x, y


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False])
def test_builds_roundtrips_with_mutable_values(full_sig: bool, partial: bool):
    # tests mutable user-specified value and default value
    result = instantiate(
        builds(f, x=[1], populate_full_signature=full_sig, hydra_partial=partial)
    )
    if partial:
        result = result()
    assert result == (
        [1],
        dict(a=2),
    )


class LocalClass:
    pass


def local_function():
    pass


@pytest.mark.parametrize(
    "obj",
    [
        RaidenDataModule,
        local_function,
        LocalClass,
        np.array,
        np.add,
        np.ufunc,  # ufuncs work!
        Adam,
        int,
        str,
        list,
        set,
        complex,
    ],
)
def test_just_roundtrip(obj):
    assert instantiate(just(obj)) is obj
