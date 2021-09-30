# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import string

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, hydrated_dataclass, instantiate
from hydra_zen.structured_configs._implementations import is_partial_builds


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
    pop_sig=st.booleans(),
)
def test_basic_hydra_meta_behavior(
    args: tuple,
    kwargs: dict,
    hydra_meta: dict,
    hydra_partial: bool,
    pop_sig: bool,
):
    Conf = builds(
        f,
        *args,
        **kwargs,
        hydra_meta=hydra_meta,
        hydra_partial=hydra_partial,
        populate_full_signature=pop_sig
    )

    conf = Conf()

    # ensure all meta-fields are present
    for meta_name, meta_val in hydra_meta.items():
        assert getattr(conf, meta_name) == meta_val

    if not hydra_partial:
        out_args, out_kwargs = instantiate(Conf)
    else:
        assert is_partial_builds(Conf)
        out_args, out_kwargs = instantiate(Conf)()

    assert out_args == args
    assert out_kwargs == kwargs
    assert set(out_kwargs).isdisjoint(hydra_meta)


def test_hydra_meta_via_hydrated_dataclass():
    @hydrated_dataclass(dict, hydra_meta=dict(a=1))
    class Conf:
        b: int = 2

    conf = Conf()
    assert conf.a == 1
    assert conf.b == 2
    assert instantiate(conf) == dict(b=2)


def test_mutable_meta_value_gets_wrapped():
    Conf = builds(int, hydra_meta=dict(a=[1, 2]))
    conf1 = Conf()
    assert conf1.a == [1, 2]
    conf1.a.append(3)

    conf2 = Conf()
    assert conf2.a == [1, 2]


def f2(*, x):
    return x


def test_deletion_by_inheritance():
    Conf = builds(dict, not_compat_with_f=1, x=3)

    with pytest.raises(TypeError):
        # presence of `not_compat_with_f` should raise
        builds(f2, builds_bases=(Conf,))

    out = instantiate(
        builds(
            f2,
            # `not_compat_with_f` should no longer be flagged
            # by our signature verification since it is now
            # a meta field.
            # We have effectively "deleted" `not_compat_with_f`
            # from this config.
            hydra_meta=dict(not_compat_with_f=None),
            builds_bases=(Conf,),
        )
    )
    assert out == 3
