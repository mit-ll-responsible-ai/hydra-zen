# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from collections import Counter
from dataclasses import dataclass, is_dataclass
from itertools import zip_longest

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given

from hydra_zen import builds, hydrated_dataclass, instantiate, just


def test_builds_with_populate_sig_raises_on_target_without_sig():
    with pytest.raises(ValueError):
        builds(dict, a=1, b="x", populate_full_signature=True)


def test_builds_returns_a_dataclass_type():
    conf = builds(dict, x=1, y="hi")
    assert is_dataclass(conf) and isinstance(conf, type)


def test_builds_hydra_partial_raises_if_recursion_disabled():
    with pytest.raises(ValueError):
        builds(dict, hydra_partial=True, hydra_recursive=False)


def x_is_var_pos(*x):
    pass


def x_is_not_in_sig(y):
    pass


def sig_is_empty():
    pass


@pytest.mark.parametrize("func", [x_is_var_pos, x_is_not_in_sig, sig_is_empty])
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_user_specified_arg_is_not_in_sig(func, full_sig, partial):
    with pytest.raises(TypeError):
        builds(func, x=10, hydra_partial=partial, populate_full_signature=full_sig)


@dataclass
class A:
    x: int = 1  # `x` is not a parameter in y


def f(y):
    return y


@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_base_has_invalid_arg(full_sig, partial):

    with pytest.raises(TypeError):
        builds(
            f,
            hydra_partial=partial,
            populate_full_signature=full_sig,
            builds_bases=(A,),
        )


@pytest.mark.parametrize(
    "target",
    [
        list,
        builds,
        just,
        Counter,
        zip_longest,
        dataclass,
        x_is_var_pos,
        sig_is_empty,
        given,
        assume,
        hydrated_dataclass,
        inspect.signature,
    ],
)
@given(partial=st.booleans(), full_sig=st.booleans())
def test_fuzz_build_validation_against_a_bunch_of_common_objects(
    target, partial: bool, full_sig: bool
):
    doesnt_have_sig = False
    try:
        inspect.signature(target)
    except ValueError:
        doesnt_have_sig = True

    if doesnt_have_sig and full_sig:
        assume(False)
    builds(target, hydra_partial=partial, populate_full_signature=full_sig)


def f2():
    pass


@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_base_with_partial_target_is_specified(
    partial: bool, full_sig: bool
):

    partiald_conf = builds(f2, hydra_partial=True)

    if not partial:
        with pytest.raises(TypeError):
            builds(
                f2,
                populate_full_signature=full_sig,
                hydra_partial=partial,
                builds_bases=(partiald_conf,),
            )
    else:
        builds(
            f2,
            populate_full_signature=full_sig,
            hydra_partial=partial,
            builds_bases=(partiald_conf,),
        )


def func_with_var_kwargs(**x):
    return x


def test_builds_validation_is_relaxed_by_presence_of_var_kwargs():
    # tests that `builds` does not flag `x` for matching the name of
    # x in the sig of `f2` since it is a var-kwarg
    #
    # tests that un-specified arg in sig is okay because of var-kwargs
    assert instantiate(builds(func_with_var_kwargs, x=2, y=10)) == {"x": 2, "y": 10}


class Class:
    pass


@pytest.mark.parametrize("not_callable", [1, "a", None, [1, 2], Class()])
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_on_non_callable_target(not_callable, partial, full_sig):
    with pytest.raises(TypeError):
        builds(not_callable, populate_full_signature=full_sig, hydra_partial=partial)


@pytest.mark.parametrize(
    "param_name, value",
    [
        ("populate_full_signature", None),
        ("hydra_recursive", None),
        ("hydra_partial", None),
        ("hydra_convert", None),
        ("hydra_convert", "wrong value"),
        ("dataclass_name", 1),
        ("builds_bases", (Class,)),
    ],
)
def test_builds_input_validation(param_name: str, value):
    def f(**kwargs):
        pass  # use **kwargs to ensure that signature checking isn't causing the raise

    builds_args = {param_name: value}
    with pytest.raises((ValueError, TypeError)):
        builds(Class, **builds_args)


def test_just_raises_with_legible_message():
    with pytest.raises(AttributeError) as exec_info:
        just(1)
    assert "just(1)" in str(exec_info.value)


def test_hydrated_dataclass_from_instance_raise():
    @dataclass
    class A:
        x: int = 1

    instance_of_a = A()
    with pytest.raises(NotImplementedError):
        hydrated_dataclass(dict)(instance_of_a)


@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_for_unimportable_target(partial, full_sig):
    def unreachable():
        pass

    with pytest.raises(ModuleNotFoundError):
        builds(unreachable, hydra_partial=partial, populate_full_signature=full_sig)


def test_just_raises_for_unimportable_target():
    def unreachable():
        pass

    with pytest.raises(ModuleNotFoundError):
        just(unreachable)
