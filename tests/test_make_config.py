# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import string
from enum import Enum
from itertools import chain
from typing import Any, Callable, Tuple, Union, get_type_hints

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, example, given, note, settings
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException, ValidationError

from hydra_zen import ZenField, builds, instantiate, make_config, to_yaml
from hydra_zen.structured_configs._make_config import NOTHING
from tests import everything_except
from tests.custom_strategies import partitions

not_a_string = everything_except(str)


def test_NOTHING_cannot_be_instantiated():
    with pytest.raises(TypeError):
        NOTHING()


@settings(max_examples=10, suppress_health_check=[HealthCheck(3)])
@given(not_a_string)
def test_validate_ZenField_name(not_str):
    with pytest.raises(TypeError):
        ZenField(name=not_str)


def test_future_compatibility_name_squatting():
    with pytest.raises(ValueError):
        make_config(hydra_zoo=1)

    with pytest.raises(ValueError):
        make_config("_zen_field")


def test_zen_field_args_positional_ordering():
    field = ZenField(int, 1, "name")
    assert field.name == "name"
    assert field.hint is int
    assert field.default.default == 1  # type: ignore


@settings(max_examples=20, suppress_health_check=[HealthCheck(3)])
@given(
    args=st.lists(not_a_string | st.just(ZenField()), min_size=1),
    as_ZenField=st.booleans(),
)
def test_validate_pos_arg_field_names(args: list, as_ZenField: bool):
    """
    Tests:
    - args as not-strings
    - args as ZenFields without names"""
    with pytest.raises((TypeError, ValueError)):
        if as_ZenField:
            args = [
                ZenField(name=name) if isinstance(name, str) else name for name in args
            ]
        make_config(*args)


@given(
    # list must have at least one redundant entry
    args=st.text(string.ascii_lowercase[:3], min_size=2)
    .filter(lambda x: len(set(x)) < len(x))
    .map(list),
    args_as_fields=st.booleans(),
    kwargs_as_fields=st.booleans(),
)
def test_validate_redundant_args(
    args: list, args_as_fields: bool, kwargs_as_fields: bool
):
    kwargs = {args[-1]: 1}
    if kwargs_as_fields:
        {a: ZenField(name=a, default=v) for a, v in kwargs.items()}

    args = args[:-1]
    if args_as_fields:
        args = [ZenField(name=a) for a in args]

    with pytest.raises(ValueError):
        make_config(*args, **kwargs)


@given(
    st.lists(
        st.sampled_from(string.ascii_lowercase), min_size=2, max_size=2, unique=True
    ).map(lambda x: (x[0], ZenField(name=x[1])))
)
def test_validate_conflicting_kwarg_name(name_and_field: Tuple[str, ZenField]):
    name, misnamed_field = name_and_field
    with pytest.raises(ValueError):
        make_config(**{name: misnamed_field})


def test_to_yaml_validates_hydra_compat_types():
    from dataclasses import dataclass
    from typing import Callable, Union

    @dataclass
    class A:
        x: Union[Callable, int]  # not supported by omegaconf

    with pytest.raises(OmegaConfBaseException):
        to_yaml(A)


@example(int)
@example(str)
@example(bool)
@example(builds(int))
@example(Tuple[int, str, bool])
@example(Callable[[int], int])
@example(Union[Callable, int])
@given(hint=st.from_type(type))
def test_types_are_sanitized(hint):
    # `to_yaml` would raise if types weren't sanitized
    to_yaml(make_config(ZenField(name="a", hint=hint)))
    to_yaml(make_config(a=ZenField(hint=hint)))


def test_type_broadening_for_builds_default():

    instantiate(make_config(a=ZenField(hint=int, default=builds(int))))

    with pytest.raises(OmegaConfBaseException):
        instantiate(
            make_config(
                a=ZenField(hint=int, default=builds(int)), hydra_recursive=False
            )
        )


class InputType(Enum):
    kwargs = 1
    pos_args = 2
    kwarg_ZenField = 3


@given(input_type=st.from_type(InputType))
def test_hydra_type_validation_works(input_type):
    if input_type in {InputType.kwargs, InputType.kwarg_ZenField}:
        Conf = make_config(x=ZenField(int))
    elif input_type is InputType.pos_args:
        Conf = make_config(ZenField(name="x", hint=int))
    else:
        assert False

    instantiate(Conf, x=1)
    with pytest.raises(ValidationError):
        instantiate(Conf, x="hi")


@settings(max_examples=500, deadline=None)
@given(
    default=st.none() | st.booleans()
    # avoid issues with interpolated fields and missing values
    | st.text().filter(lambda x: "${" not in x and "?" not in x)
    | st.lists(st.booleans())
    | st.dictionaries(st.booleans(), st.booleans())
    | st.just(print)
    | st.builds(int),
    input_type=st.from_type(InputType),
)
def test_default_values_get_set_as_expected(default, input_type):
    if input_type is InputType.kwargs:
        Conf = make_config(a=default)
    elif input_type is InputType.kwarg_ZenField:
        Conf = make_config(a=ZenField(default=default))
    elif input_type is InputType.pos_args:
        Conf = make_config(ZenField(name="a", default=default))
    else:
        assert False

    instantiate(OmegaConf.create(to_yaml(Conf))).a == default


@given(
    input_type=st.from_type(InputType),
)
def test_mutable_default_value_uses_default_factory(input_type):
    default = [1, 2, 3]

    if input_type is InputType.kwargs:
        Conf = make_config(a=default)
    elif input_type is InputType.kwarg_ZenField:
        Conf = make_config(a=ZenField(default=default))
    elif input_type is InputType.pos_args:
        Conf = make_config(ZenField(name="a", default=default))
    else:
        assert False

    x = Conf()
    y = Conf()
    x.a.append(-100)  # shouldn't mutate things
    z = Conf()
    assert x.a == [1, 2, 3, -100]
    assert y.a == [1, 2, 3]
    assert z.a == [1, 2, 3]


@settings(deadline=None)
@given(
    args_kwargs=partitions(
        st.lists(st.sampled_from(string.ascii_lowercase), unique=True)
    ),
    data=st.data(),
)
def test_all_fields_get_set(args_kwargs, data: st.DataObject):
    """Tests that fields with and without defaults get set to
    the resulting config.
    Ensures:
      - fields get re-ordered so that fields without defaults precede those that do
      - resulting config is yaml-serializable
    """
    # disjoint collections of valid field names
    args, kwargs = args_kwargs

    # In this test: default value of None does not get set
    # Randomly draw/assign default (or not) for each field
    default_args = data.draw(st.tuples(*[st.sampled_from([None, 1, 2])] * len(args)))
    default_kwargs = data.draw(
        st.tuples(*[st.sampled_from([None, 1, 2])] * len(kwargs))
    )

    # Assign defaults to those fields who have them
    args = [
        a if v is None else ZenField(name=a, default=v)
        for a, v in zip(args, default_args)
    ]

    # Assign defaults to those fields who have them
    kwargs = {
        k: (ZenField() if v is None else ZenField(default=v))
        for k, v in zip(kwargs, default_kwargs)
    }

    Conf = make_config(*args, **kwargs)
    sig = inspect.signature(Conf).parameters

    for name, default in zip(chain(*args_kwargs), chain(default_args, default_kwargs)):
        if default is None:
            assert not hasattr(Conf, name)  # no default value
        else:
            assert getattr(Conf, name) == default
        assert name in sig

    to_yaml(Conf)  # ensure serializable


@given(name=st.sampled_from(["foo", "bar"]))
def test_named_config(name):
    assert make_config(zen_dataclass={"cls_name": name}).__name__ == name


def test_bases():
    A = make_config()
    B = make_config()
    C = make_config(bases=(A,))
    assert issubclass(C, A)
    assert not issubclass(C, B)


@given(
    hydra_convert=st.none() | st.sampled_from(["all", "partial", "none"]),
    hydra_recursive=st.none() | st.booleans(),
)
def test_hydra_options(hydra_convert, hydra_recursive):
    Builds = builds(dict, hydra_convert=hydra_convert, hydra_recursive=hydra_recursive)
    Conf = make_config(hydra_convert=hydra_convert, hydra_recursive=hydra_recursive)

    b_yaml = to_yaml(Builds)
    c_yaml = to_yaml(Conf)
    note(b_yaml)
    note(c_yaml)
    if hydra_convert is None and hydra_recursive is None:
        assert c_yaml == "{}\n"  # yaml for empty config
    else:
        assert b_yaml.splitlines()[1:] == c_yaml.splitlines()


def test_make_config_annotation_widening_behavior():
    BuildsInt = builds(int)
    Conf1 = make_config(a=ZenField(hint=BuildsInt, default=BuildsInt()))
    assert get_type_hints(Conf1)["a"] is BuildsInt

    Conf2 = make_config(b=ZenField(hint=int, default=BuildsInt()))
    assert get_type_hints(Conf2)["b"] is Any
