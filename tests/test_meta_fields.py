# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import string

import hypothesis.strategies as st
from hypothesis import given

from hydra_zen import builds, instantiate
from hydra_zen.structured_configs._implementations import is_partial_builds
from hydra_zen.typing import Builds, PartialBuilds


def f(*args, **kwargs):
    return args, kwargs


@given(
    args=st.tuples(st.integers()),
    kwargs=st.dictionaries(
        st.text(string.ascii_letters, min_size=1, max_size=1),
        st.integers(),
    ),
    hydra_meta=st.dictionaries(
        st.text(string.ascii_letters, min_size=2, max_size=2).filter(
            lambda x: x not in {"is", "in", "as", "or", "if"}  # reserved fields
        ),
        st.integers(),
    ),
    hydra_partial=st.booleans(),
)
def test_basic_meta_behavior(
    args: tuple, kwargs: dict, hydra_meta: dict, hydra_partial: bool
):
    Conf = builds(
        f, *args, **kwargs, hydra_meta=hydra_meta, hydra_partial=hydra_partial
    )

    conf = Conf()
    for meta_name, meta_val in hydra_meta.items():
        assert getattr(conf, meta_name) == meta_val

    if not hydra_partial:
        out_args, out_kwargs = instantiate(Conf)
    else:
        assert is_partial_builds(Conf)
        out_args, out_kwargs = instantiate(Conf)()

    assert out_args == args
    assert out_kwargs == kwargs
    assert set(kwargs).isdisjoint(hydra_meta)
