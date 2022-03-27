# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, is_dataclass
from itertools import zip_longest

import hypothesis.strategies as st
import pytest
from hydra import __version__ as HYDRA_VERSION
from hypothesis import assume, given, settings
from omegaconf import OmegaConf

from hydra_zen import builds, get_target, hydrated_dataclass, instantiate, just, to_yaml
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.structured_configs._globals import (
    HYDRA_FIELD_NAMES,
    ZEN_TARGET_FIELD_NAME,
)
from hydra_zen.structured_configs._utils import get_obj_path
from tests import everything_except


def test_builds_no_args_raises():
    with pytest.raises(TypeError):
        builds()  # type: ignore


def test_builds_no_positional_target_raises():
    with pytest.raises(TypeError):
        builds(hydra_target=dict)  # type: ignore


def test_target_as_kwarg_is_an_error():
    with pytest.raises(TypeError):
        builds(target=int)  # type: ignore


def test_hydra_partial_is_error():
    with pytest.raises(ValueError):
        builds(int, hydra_partial=True)


def test_hydra_partial_via_hydrated_dataclass_is_error():
    with pytest.raises(TypeError):

        @hydrated_dataclass(int, hydra_partial=True)  # type: ignore
        class A:
            pass


@given(hydra_partial=st.booleans(), zen_partial=st.booleans())
def test_specifying_hydra_partial_and_zen_partial_raises(
    hydra_partial: bool, zen_partial: bool
):
    with pytest.raises(ValueError):
        builds(int, hydra_partial=hydra_partial, zen_partial=zen_partial)


@given(hydra_partial=st.booleans(), zen_partial=st.just(True))
def test_specifying_hydra_partial_and_zen_partial_raises_in_hydrated(
    hydra_partial: bool, zen_partial: bool
):
    with pytest.raises(TypeError):

        @hydrated_dataclass(int, hydra_partial=hydra_partial, zen_partial=zen_partial)  # type: ignore
        class A:
            pass


def test_deprecation_shim_for_hydrated_dataclass_doesnt_permit_new_kwargs():
    with pytest.raises(TypeError):

        @hydrated_dataclass(int, some_arg=1)  # type: ignore
        class A:
            pass


@pytest.mark.skipif(
    HYDRA_VERSION < "1.1.1", reason="Hydra squatted on the name 'target' until v1.1.1"
)
def test_builds_with_target_as_named_arg_works():
    out = instantiate(builds(dict, target=1, b=2))
    assert out == {"target": 1, "b": 2}


def test_builds_with_populate_sig_raises_on_target_without_sig():
    with pytest.raises(ValueError):
        builds(dict, a=1, b="x", populate_full_signature=True)


def test_builds_returns_a_dataclass_type():
    conf = builds(dict, x=1, y="hi")
    assert is_dataclass(conf) and isinstance(conf, type)


@settings(deadline=None)
@given(everything_except(Mapping, type(None)))
def test_builds_zen_meta_not_mapping_raises(not_a_mapping):
    with pytest.raises(TypeError):
        builds(int, zen_meta=not_a_mapping)


def test_builds_zen_meta_with_non_string_keys_raises():
    with pytest.raises(TypeError):
        builds(int, zen_meta={1: None})  # type: ignore


def f_starx(*x):
    pass


def f_kwargs(**kwargs):
    pass


def f_y(y):
    pass


def f_empty():
    pass


def f_x_y2_str_z3(x, y=2, *, z=3):
    pass


def passthrough(*args, **kwargs):
    pass


@pytest.mark.parametrize(
    "func, args, kwargs",
    [
        # named param not in sig
        (f_starx, (), dict(x=10)),
        (f_starx, (), dict(y=10)),
        (f_y, (), dict(x=10)),
        (f_empty, (), dict(x=10)),
        # too many pos args
        (f_kwargs, (1, 2), dict(y=2)),
        (f_x_y2_str_z3, (1, 2, 3), {}),
        (f_empty, (1,), {}),
        (f_y, (1, 2), {}),
        # multiple values specified for param
        (f_y, (1,), dict(y=1)),
        (
            f_x_y2_str_z3,
            (1, 2),
            dict(y=1, z=4),
        ),
    ],
)
@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_user_specified_args_violate_sig(
    func, args, kwargs, full_sig, partial
):
    with pytest.raises(TypeError):
        builds(
            func,
            *args,
            **kwargs,
            zen_partial=partial,
            populate_full_signature=full_sig,
        )

    # test when **kwargs are inherited
    kwarg_base = builds(passthrough, **kwargs, zen_partial=partial)
    with pytest.raises(TypeError):
        builds(
            func,
            *args,
            zen_partial=partial,
            populate_full_signature=full_sig,
            builds_bases=(kwarg_base,),
        )
    del kwarg_base

    # test when *args are inherited
    args_base = builds(passthrough, *args, zen_partial=partial)
    with pytest.raises(TypeError):
        builds(
            func,
            **kwargs,
            zen_partial=partial,
            populate_full_signature=full_sig,
            builds_bases=(args_base,),
        )
    del args_base

    # test when *args and **kwargs are inherited
    args_kwargs_base = builds(passthrough, *args, **kwargs, zen_partial=partial)
    with pytest.raises(TypeError):
        builds(
            func,
            zen_partial=partial,
            populate_full_signature=full_sig,
            builds_bases=(args_kwargs_base,),
        )
    del args_kwargs_base


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
            zen_partial=partial,
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
        f_starx,
        f_empty,
        given,
        assume,
        hydrated_dataclass,
        inspect.signature,
    ],
)
@given(full_sig=st.booleans())
def test_fuzz_build_validation_against_a_bunch_of_common_objects(
    target, full_sig: bool
):
    doesnt_have_sig = False
    try:
        inspect.signature(target)
    except ValueError:
        doesnt_have_sig = True

    if doesnt_have_sig and full_sig:
        assume(False)

    conf = builds(target, zen_partial=True, populate_full_signature=full_sig)

    OmegaConf.create(to_yaml(conf))  # ensure serializable
    instantiate(conf)  # ensure instantiable


def f2():
    pass


@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_when_base_with_partial_target_is_specified(
    partial: bool, full_sig: bool
):

    partiald_conf = builds(f2, zen_partial=True)

    if not partial:
        with pytest.raises(TypeError):
            builds(
                f2,
                populate_full_signature=full_sig,
                zen_partial=partial,
                builds_bases=(partiald_conf,),
            )
    else:
        builds(
            f2,
            populate_full_signature=full_sig,
            zen_partial=partial,
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
        builds(not_callable, populate_full_signature=full_sig, zen_partial=partial)


@pytest.mark.parametrize(
    "param_name, value",
    [
        ("populate_full_signature", None),
        ("hydra_recursive", 1),
        ("zen_partial", 1),
        ("hydra_convert", 1),
        ("hydra_convert", "wrong value"),
        ("dataclass_name", 1),
        ("builds_bases", (Class,)),
        ("frozen", 1),
    ],
)
def test_builds_input_validation(param_name: str, value):
    def f(**kwargs):
        pass  # use **kwargs to ensure that signature checking isn't causing the raise

    builds_args = {param_name: value}
    with pytest.raises((ValueError, TypeError)):
        builds(Class, **builds_args)


def test_just_raises_with_legible_message():
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(Class())  # type: ignore


def test_hydrated_dataclass_from_instance_raise():
    @dataclass
    class A:
        x: int = 1

    instance_of_a = A()
    with pytest.raises(NotImplementedError):
        hydrated_dataclass(dict)(instance_of_a)  # type: ignore


@given(partial=st.booleans(), full_sig=st.booleans())
def test_builds_raises_for_unimportable_target(partial, full_sig):
    def unreachable():
        pass

    with pytest.raises(ModuleNotFoundError):
        builds(unreachable, zen_partial=partial, populate_full_signature=full_sig)


def test_just_raises_for_unimportable_target():
    def unreachable():
        pass

    with pytest.raises(ModuleNotFoundError):
        just(unreachable)


def test_get_target_on_non_builds():
    with pytest.raises(TypeError):
        get_target(1)  # type: ignore


RESERVED_FIELD_NAMES = sorted(HYDRA_FIELD_NAMES) + [
    ZEN_TARGET_FIELD_NAME,
    "_zen_some_new_feature",
    "hydra_some_new_feature",
    "zen_some_new_feature",
]


@pytest.mark.parametrize("field", RESERVED_FIELD_NAMES)
def test_reserved_names_are_reserved(field: str):
    kwargs = {field: True}
    with pytest.raises(ValueError):
        builds(dict, **kwargs)


@pytest.mark.parametrize("field", RESERVED_FIELD_NAMES)
def test_reserved_names_are_reserved_by_zen_meta(field: str):
    kwargs = {field: True}
    with pytest.raises(ValueError):
        builds(dict, zen_meta=kwargs)


def f_meta_sig(x, *args, y, **kwargs):
    pass


@given(
    meta_fields=st.dictionaries(
        st.sampled_from(["x", "y", "args", "kwargs", "z"]), st.integers()
    ),
    pop_sig=st.booleans(),
    partial=st.booleans(),
)
def test_meta_fields_colliding_with_sig_raises(
    meta_fields, pop_sig: bool, partial: bool
):
    if {"x", "y"} & set(meta_fields):
        with pytest.raises(ValueError):
            builds(
                f_meta_sig,
                zen_meta=meta_fields,
                populate_full_signature=pop_sig,
                zen_partial=partial,
            )
    else:
        builds(
            f_meta_sig,
            zen_meta=meta_fields,
            populate_full_signature=pop_sig,
            zen_partial=partial,
        )


@given(
    meta_fields=st.dictionaries(
        st.sampled_from(["x", "y", "args", "kwargs", "z"]), st.integers()
    ),
    partial=st.booleans(),
)
def test_meta_fields_colliding_with_user_provided_kwargs_raises(
    meta_fields, partial: bool
):
    if {"x", "y"} & set(meta_fields):
        with pytest.raises(ValueError):
            builds(dict, x=1, y=2, zen_meta=meta_fields, zen_partial=partial)
    else:
        builds(dict, x=1, y=2, zen_meta=meta_fields, zen_partial=partial)


def test_get_obj_path_raises_for_unknown_name():
    with pytest.raises(AttributeError):
        get_obj_path(1)
