# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import string
from typing import Any, Dict, List

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf

from hydra_zen import builds, instantiate, just, to_yaml
from tests import valid_hydra_literals

arbitrary_kwargs = st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters, min_size=1, max_size=1),
    values=valid_hydra_literals,
)


def pass_through_kwargs(**kwargs):
    return kwargs


def pass_through_args(*args):
    return args


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
def test_builds_kwargs_roundtrip_with_partial(
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


@given(
    partial_args=arbitrary_kwargs.map(lambda x: list(x.values())),
    call_args=arbitrary_kwargs.map(lambda x: list(x.values())),
    full_sig=st.booleans(),
)
def test_builds_args_roundtrip_with_partial(
    partial_args: List[Any],
    call_args: List[Any],
    full_sig: bool,
):
    partial_struct = instantiate(
        builds(
            pass_through_args,
            hydra_partial=True,
            populate_full_signature=full_sig,
            *partial_args,
        ),
    )

    expected_args = partial_args.copy()
    expected_args.extend(call_args)
    assert tuple(expected_args) == partial_struct(*call_args)  # resolve partial


def f(x, y=dict(a=2)):
    return x, y


@pytest.mark.parametrize("full_sig", [True, False])
@pytest.mark.parametrize("partial", [True, False])
@pytest.mark.parametrize("named_arg", [True, False])
def test_builds_roundtrips_with_mutable_values(
    full_sig: bool, partial: bool, named_arg: bool
):
    # tests mutable user-specified value and default value
    if named_arg:
        result = instantiate(
            builds(f, x=[1], populate_full_signature=full_sig, hydra_partial=partial)
        )
    else:
        result = instantiate(
            builds(f, [1], populate_full_signature=full_sig, hydra_partial=partial)
        )
    if partial:
        result = result()
    assert result == ([1], {"a": 2})


class LocalClass:
    pass


def local_function():
    pass


@pytest.mark.parametrize(
    "obj",
    [
        local_function,
        LocalClass,
        int,
        str,
        list,
        set,
        complex,
    ],
)
def test_just_roundtrip(obj):
    cfg = just(obj)
    assert instantiate(cfg) is obj
    assert instantiate(OmegaConf.create(to_yaml(cfg))) is obj
