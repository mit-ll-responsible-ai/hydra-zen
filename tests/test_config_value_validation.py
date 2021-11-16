import inspect
from dataclasses import dataclass
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings

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
    assume(not isinstance(a, KNOWN_MUTABLE_TYPES))

    @hydrated_dataclass(target)
    class Config:
        x: Any = a

    return Config


def make_dataclass(a):
    assume(not isinstance(a, KNOWN_MUTABLE_TYPES))

    @dataclass
    class C:
        x: Any = a

    return C


class SomeType:
    pass


unsupported_instance = SomeType()


def f_with_bad_default_value(x=unsupported_instance):
    pass


@pytest.mark.parametrize(
    "config_construction_fn",
    [
        lambda x: make_config(a=x),
        lambda x: make_config(ZenField(name="a", default=x)),
        lambda x: builds(f_concrete_sig, x=x, populate_full_signature=False),
        lambda x: builds(f_concrete_sig, x, populate_full_signature=False),
        lambda x: builds(f_with_kwargs, x=x, populate_full_signature=False),
        lambda x: builds(f_with_kwargs, x, populate_full_signature=False),
        lambda x: builds(f_concrete_sig, x=x, populate_full_signature=True),
        lambda x: builds(f_concrete_sig, x, populate_full_signature=True),
        lambda x: builds(f_with_kwargs, x=x, populate_full_signature=True),
        lambda x: builds(f_with_kwargs, x, populate_full_signature=True),
        lambda x: builds(f_with_bad_default_value, populate_full_signature=True),
        lambda x: make_hydrated_dataclass(f_concrete_sig, x),
        # test validation via inheritance
        lambda x: builds(f_concrete_sig, builds_bases=(make_dataclass(x),)),
    ],
)
@settings(max_examples=20, deadline=None)
@given(
    unsupported=st.just(unsupported_instance)
    | everything_except(
        *(HYDRA_SUPPORTED_PRIMITIVES + ZEN_SUPPORTED_PRIMITIVES)
    ).filter(lambda x: not inspect.isfunction(x))
)
def test_unsupported_config_value_raises_while_making_config(
    config_construction_fn, unsupported
):
    with pytest.raises((HydraZenUnsupportedPrimitiveError, ModuleNotFoundError)):
        config_construction_fn(unsupported)


@pytest.mark.parametrize(
    "config_construction_fn",
    [
        lambda x: make_config(a=x),
        lambda x: make_config(ZenField(name="a", default=x)),
        lambda x: builds(f_concrete_sig, x=x, populate_full_signature=False),
        lambda x: builds(f_concrete_sig, x, populate_full_signature=False),
        lambda x: builds(f_with_kwargs, x=x, populate_full_signature=False),
        lambda x: builds(f_with_kwargs, x, populate_full_signature=False),
        lambda x: builds(f_concrete_sig, x=x, populate_full_signature=True),
        lambda x: builds(f_concrete_sig, x, populate_full_signature=True),
        lambda x: builds(f_with_kwargs, x=x, populate_full_signature=True),
        lambda x: builds(f_with_kwargs, x, populate_full_signature=True),
        lambda x: builds(f_with_bad_default_value, populate_full_signature=True),
        lambda x: make_hydrated_dataclass(f_concrete_sig, x),
        # test validation via inheritance
        lambda x: builds(f_concrete_sig, builds_bases=(make_dataclass(x),)),
    ],
)
@given(value=everything_except())
def test_that_configs_passed_by_zen_validation_are_serializable(
    config_construction_fn, value
):
    try:
        Conf = config_construction_fn(value)
    except (HydraZenUnsupportedPrimitiveError, ModuleNotFoundError):
        return
    # check serializability & instantiability
    to_yaml(Conf)
    instantiate(Conf)
