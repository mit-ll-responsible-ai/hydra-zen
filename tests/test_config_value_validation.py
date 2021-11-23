# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, settings

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    to_yaml,
)
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.structured_configs._implementations import HYDRA_SUPPORTED_PRIMITIVES
from hydra_zen.structured_configs._utils import KNOWN_MUTABLE_TYPES
from hydra_zen.structured_configs._value_conversion import ZEN_SUPPORTED_PRIMITIVES
from tests import everything_except


def f_concrete_sig(x):
    pass


def f_with_kwargs(*args, **kwargs):
    pass


def make_hydrated_dataclass(target, a):
    assume(not isinstance(a, tuple(KNOWN_MUTABLE_TYPES)))

    @hydrated_dataclass(target)
    class Config:
        x: Any = a

    return Config


def make_dataclass(a):
    assume(not isinstance(a, tuple(KNOWN_MUTABLE_TYPES)))

    @dataclass
    class C:
        x: Any = a

    return C


class SomeType:
    pass


class SubclassOfSupportedPrimitive(int):
    def __repr__(self) -> str:
        return "SubclassOfSupportedPrimitive(" + super().__repr__() + ")"


unsupported_subclass = SubclassOfSupportedPrimitive()
unsupported_instance = SomeType()


def f_with_bad_default_value(x=unsupported_instance):
    pass


construction_fn_variations = [
    lambda x: make_config(a=x),
    lambda x: make_config(ZenField(name="a", default=x)),
    lambda x: builds(f_concrete_sig, x=x, populate_full_signature=False),
    lambda x: builds(f_concrete_sig, x, populate_full_signature=False),  #
    lambda x: builds(f_with_kwargs, x=x, populate_full_signature=False),
    lambda x: builds(f_with_kwargs, x, populate_full_signature=False),  #
    lambda x: builds(f_concrete_sig, x=x, populate_full_signature=True),
    lambda x: builds(f_concrete_sig, x, populate_full_signature=True),  #
    lambda x: builds(f_with_kwargs, x=x, populate_full_signature=True),
    lambda x: builds(f_with_kwargs, x, populate_full_signature=True),  #
    lambda x: builds(f_with_bad_default_value, populate_full_signature=True),
    lambda x: make_hydrated_dataclass(f_concrete_sig, x),
    # test validation via inheritance
    lambda x: builds(f_concrete_sig, builds_bases=(make_dataclass(x),)),
]


@pytest.mark.parametrize(
    "config_construction_fn",
    construction_fn_variations,
)
@example(unsupported=unsupported_instance)
# test collections containing unsupported values
@example(unsupported=[unsupported_instance])
@example(unsupported=(unsupported_instance,))
@example(unsupported={unsupported_instance: 1})
@example(unsupported={1: unsupported_instance})
@example(unsupported={unsupported_instance})
@example(unsupported={unsupported_instance})
@example(unsupported=unsupported_subclass)
@example(unsupported=[unsupported_subclass])
@example(unsupported=(unsupported_subclass,))
@example(unsupported={unsupported_subclass: 1})
@example(unsupported={1: unsupported_subclass})
@example(unsupported={unsupported_subclass})
@example(unsupported={unsupported_subclass})
# Hydra doesn't support dataclass nodes for keys; ensure
# hydra-zen doesn't provide enhanced primitive support for keys
@example(unsupported={1j: 1})
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=(HealthCheck.data_too_large, HealthCheck.too_slow),
)
@given(
    unsupported=everything_except(
        *(HYDRA_SUPPORTED_PRIMITIVES | ZEN_SUPPORTED_PRIMITIVES)
    ).filter(lambda x: not inspect.isfunction(x))
)
def test_unsupported_config_value_raises_while_making_config(
    config_construction_fn, unsupported
):
    with pytest.raises((HydraZenUnsupportedPrimitiveError, ModuleNotFoundError)):
        config_construction_fn(unsupported)


@pytest.mark.parametrize("config_construction_fn", construction_fn_variations)
@settings(
    suppress_health_check=(HealthCheck.data_too_large, HealthCheck.too_slow),
    deadline=None,
)
@given(value=everything_except())
def test_that_configs_passed_by_zen_validation_are_serializable(
    config_construction_fn, value
):
    # `value` is literally any value that Hypothesis knows how to describe
    try:
        Conf = config_construction_fn(value)
    except (HydraZenUnsupportedPrimitiveError, ModuleNotFoundError):
        # the drawn value is not compatible with Hydra -- should be caught
        # by us
        return
    # The value passed our construction, thus the resulting config
    # should be serializable & instantiable
    to_yaml(Conf)
    instantiate(Conf)
