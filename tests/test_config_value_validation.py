# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from dataclasses import dataclass
from typing import Any

import pytest
from hypothesis import HealthCheck, assume, example, given, note, settings
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import KeyValidationError

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    to_yaml,
)
from hydra_zen._compatibility import ZEN_SUPPORTED_PRIMITIVES
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.structured_configs._implementations import HYDRA_SUPPORTED_PRIMITIVES
from tests import everything_except


def f_concrete_sig(x):
    pass


def f_with_kwargs(*args, **kwargs):
    pass


def make_hydrated_dataclass(target, a):
    assume(a.__hash__ is not None)

    @hydrated_dataclass(target)
    class Config:
        x: Any = a

    return Config


def make_dataclass(a):
    assume(a.__hash__ is not None)

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


def test_omegaconf_doesnt_permit_dataclasses_as_dict_keys():
    C = builds(int, 1, zen_dataclass={"frozen": True})
    instantiate(OmegaConf.create({1: C}))  # should be OK

    with pytest.raises(KeyValidationError):
        OmegaConf.create(OmegaConf.create({C: 1}))


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
    lambda _: builds(f_with_bad_default_value, populate_full_signature=True),
    lambda x: make_hydrated_dataclass(f_concrete_sig, x),
    # test validation via inheritance
    lambda x: builds(f_concrete_sig, builds_bases=(make_dataclass(x),)),
    # test validation of meta-fields
    lambda x: builds(f_concrete_sig, x=1, zen_meta=dict(a=x)),
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
@example(unsupported=unsupported_subclass)
@example(unsupported=[unsupported_subclass])
@example(unsupported=(unsupported_subclass,))
@example(unsupported={unsupported_subclass: 1})
@example(unsupported={1: unsupported_subclass})
@example(unsupported={unsupported_subclass})
# Hydra doesn't support dataclass nodes for keys; ensure
# hydra-zen doesn't provide enhanced primitive support for keys
@example(unsupported={make_dataclass(1): 1})
@example(unsupported={SomeType: 1})
@example(unsupported={1j: 1})
@example(unsupported={ListConfig([1, 2]): 1})
@example(unsupported={DictConfig({1: 1}): 1})
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
    note(f"unsupported: {unsupported}")
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
    if isinstance(value, str):
        # avoid interpolation issues
        assume(not value.startswith("${"))

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
