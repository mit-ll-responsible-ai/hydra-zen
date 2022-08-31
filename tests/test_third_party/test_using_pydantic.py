# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import dataclasses
from typing import List, Optional

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from pydantic import AnyUrl, Field, PositiveFloat
from pydantic.dataclasses import dataclass as pyd_dataclass
from typing_extensions import Literal

from hydra_zen import builds, instantiate, just
from hydra_zen.third_party.pydantic import validates_with_pydantic

parametrize_pydantic_fields = pytest.mark.parametrize(
    "custom_type, good_val, bad_val",
    [
        (PositiveFloat, 22, -1),
        (AnyUrl, "http://www.pythonlikeyoumeanit.com", "hello"),
    ],
)


@parametrize_pydantic_fields
def test_pydantic_specific_fields_function(custom_type, good_val, bad_val):
    def f(x):
        return x

    f.__annotations__["x"] = custom_type
    wrapped = validates_with_pydantic(f)

    wrapped(good_val)  # ok
    with pytest.raises(Exception):
        wrapped(bad_val)


@parametrize_pydantic_fields
def test_pydantic_specific_fields_class(custom_type, good_val, bad_val):
    class A:
        def __init__(self, x) -> None:
            pass

    A.__init__.__annotations__["x"] = custom_type
    validates_with_pydantic(A)

    A(good_val)
    with pytest.raises(Exception):
        A(bad_val)


def test_custom_validation_config():
    # test that users can pass a custom-configured instance of
    # `pydantic.validate_arguments`
    from functools import partial

    from pydantic import validate_arguments

    class A:
        pass

    def f(x: A):
        return x

    yes_arb_types = partial(
        validates_with_pydantic,
        validator=validate_arguments(config=dict(arbitrary_types_allowed=True)),
    )

    no_arb_types = partial(
        validates_with_pydantic,
        validator=validate_arguments(config=dict(arbitrary_types_allowed=False)),
    )

    yes_arb_types(f)(A())

    with pytest.raises(RuntimeError):
        no_arb_types(f)(A())


@pyd_dataclass
class PydanticConf:
    x: Literal[1, 2]
    y: int = 2


@pytest.mark.parametrize("x", [1, 2])
def test_documented_example_passes(x):
    HydraConf = builds(PydanticConf, populate_full_signature=True)
    conf = instantiate(HydraConf, x=x)
    assert isinstance(conf, PydanticConf)
    assert conf == PydanticConf(x=x, y=2)


@settings(max_examples=20)
@given(x=(st.integers() | st.floats()).filter(lambda x: x != 1 and x != 2))
def test_documented_example_raises(x):
    HydraConf = builds(PydanticConf, populate_full_signature=True)
    with pytest.raises(Exception):
        # using a broad exception here because of
        # re-raising incompatibilities with Hydra
        instantiate(HydraConf, x=x)


@pyd_dataclass
class User:
    id: int
    name: str = "John Doe"
    friends: List[int] = dataclasses.field(default_factory=lambda: [0])
    age: Optional[int] = dataclasses.field(
        default=None,
        metadata=dict(title="The age of the user", description="do not lie!"),
    )

    # TODO: add support for pydantic.Field
    height: Optional[int] = Field(None, title="The height in cm", ge=50, le=300)


@pytest.mark.parametrize(
    "config_maker",
    [
        lambda u1: just(u1, hydra_convert="all"),
        # test hydra-convert applies within container
        lambda u1: just([u1], hydra_convert="all")[0],
    ],
)
def test_just_on_pydantic_dataclass(config_maker):
    u1 = User(id="42")  # type: ignore
    Conf = config_maker(u1)
    ju1 = instantiate(Conf)
    assert ju1 == u1 and isinstance(ju1, User)


# def test_pydantic_runtime_type_checking():
#     from hydra.errors import InstantiationException
#     Conf = builds(User, populate_full_signature=True)
#     inst_bad = Conf(id=22, height=10, age=-100)
#     inst_good = Conf(id=22, height=10, age=25)
#     with pytest.raises(InstantiationException):
#         instantiate(inst_bad)
#     assert inst_good == User(id=22, height=10, age=25) and isinstance(inst_good, User)
