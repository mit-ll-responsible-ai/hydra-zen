# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import string

import hypothesis.strategies as st
import pytest
from hypothesis import given

from hydra_zen import builds, hydrated_dataclass, instantiate
from hydra_zen.structured_configs._type_guards import is_partial_builds


def f(*args, **kwargs):
    return args, kwargs


@given(
    args=st.tuples(st.integers()),
    kwargs=st.dictionaries(
        st.text(string.ascii_letters, min_size=1, max_size=1),
        st.integers(),
    ),
    zen_meta=st.dictionaries(
        st.text(string.ascii_letters, min_size=2, max_size=2).filter(
            lambda x: x not in {"is", "in", "as", "or", "if"}  # reserved fields
        ),
        st.integers(),
    ),
    zen_partial=st.none() | st.booleans(),
    pop_sig=st.booleans(),
)
def test_basic_zen_meta_behavior(
    args: tuple,
    kwargs: dict,
    zen_meta: dict,
    zen_partial: bool,
    pop_sig: bool,
):
    Conf = builds(
        f,
        *args,
        **kwargs,
        zen_meta=zen_meta,
        zen_partial=zen_partial,
        populate_full_signature=pop_sig,
    )

    conf = Conf()

    # ensure all meta-fields are present
    for meta_name, meta_val in zen_meta.items():
        assert getattr(conf, meta_name) == meta_val

    if not zen_partial:
        out_args, out_kwargs = instantiate(Conf)
    else:
        assert is_partial_builds(Conf)
        out_args, out_kwargs = instantiate(Conf)()

    assert out_args == args
    assert out_kwargs == kwargs
    assert set(out_kwargs).isdisjoint(zen_meta)


def f_concrete_sig(a, b):
    return (a, b)


@given(
    a=st.integers(),
    b=st.integers(),
    as_kwarg=st.booleans(),
    zen_meta=st.dictionaries(
        st.text(string.ascii_letters, min_size=1, max_size=1).filter(
            lambda x: x not in {"a", "b"}
        ),
        st.integers(),
    ),
    zen_partial=st.none() | st.booleans(),
    pop_sig=st.booleans(),
)
def test_basic_zen_meta_behavior_vs_concrete_sig(
    a: int,
    b: int,
    as_kwarg: bool,
    zen_meta: dict,
    zen_partial: bool,
    pop_sig: bool,
):
    if as_kwarg:
        args = ()
        kwargs = dict(a=a, b=b)
    else:
        args = (a, b)
        kwargs = {}

    Conf = builds(
        f_concrete_sig,
        *args,
        **kwargs,
        zen_meta=zen_meta,
        zen_partial=zen_partial,
        populate_full_signature=pop_sig,
    )

    conf = Conf()  # type: ignore

    # ensure all meta-fields are present
    for meta_name, meta_val in zen_meta.items():
        assert getattr(conf, meta_name) == meta_val

    if not zen_partial:
        out_args = instantiate(Conf)
    else:
        assert is_partial_builds(Conf)
        out_args = instantiate(Conf)()

    assert out_args == (a, b)


def test_zen_meta_via_hydrated_dataclass():
    @hydrated_dataclass(dict, zen_meta=dict(a=1))
    class Conf:
        b: int = 2

    conf = Conf()
    assert conf.a == 1  # type: ignore
    assert conf.b == 2
    assert instantiate(conf) == dict(b=2)


def test_mutable_meta_value_gets_wrapped():
    Conf = builds(int, zen_meta=dict(a=[1, 2]))
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
            zen_meta=dict(not_compat_with_f=None),
            builds_bases=(Conf,),
        )
    )
    assert out == 3
