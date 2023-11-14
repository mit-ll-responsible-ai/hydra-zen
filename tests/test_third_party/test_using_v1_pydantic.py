# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import dataclasses
from typing import Any, List, Optional

import hypothesis.strategies as st
import pydantic
import pytest
from hydra.errors import InstantiationException
from hypothesis import given, settings
from omegaconf import OmegaConf
from pydantic import AnyUrl, BaseModel, Field, PositiveFloat
from pydantic.dataclasses import dataclass as pyd_dataclass
from typing_extensions import Literal

from hydra_zen import builds, get_target, instantiate, just, to_yaml
from hydra_zen.third_party.pydantic import validates_with_pydantic

if pydantic.__version__.startswith("2."):
    pytest.skip("These tests are for pydantic v1", allow_module_level=True)

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


class BaseModelConf(BaseModel):
    x: Literal[1, 2]
    y: int = 2


@pytest.mark.parametrize("Target", [PydanticConf, BaseModelConf])
@pytest.mark.parametrize("x", [1, 2])
def test_documented_example_passes(Target, x):
    HydraConf = builds(Target, populate_full_signature=True)
    conf = instantiate(HydraConf, x=x)
    assert isinstance(conf, Target)
    assert conf == Target(x=x, y=2)
    assert get_target(HydraConf) is Target


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


def test_pydantic_runtime_type_checking():
    Conf = builds(User, populate_full_signature=True, hydra_convert="all")
    inst_bad = Conf(id=22, height=-50, age=-100)
    with pytest.raises(InstantiationException):
        instantiate(inst_bad)  # pydantic raises on invalid height

    inst_good = instantiate(Conf(id=22, height=50, age=25))
    assert inst_good == User(id=22, height=50, age=25) and isinstance(inst_good, User)


@pyd_dataclass
class HasDefault:
    x: int = Field(default=1)


@pyd_dataclass
class HasDefaultFactory:
    x: Any = Field(default_factory=lambda: [1 + 2j])


class BaseModelHasDefault(BaseModel):
    x: int = Field(default=1)


class BaseModelHasDefaultFactory(BaseModel):
    x: Any = Field(default_factory=lambda: [1 + 2j])


@pytest.mark.parametrize(
    "target,kwargs",
    [
        (HasDefault, {}),
        (BaseModelHasDefault, {}),
        (HasDefaultFactory, {}),
        (BaseModelHasDefaultFactory, {}),
        (HasDefault, {"x": 12}),
        (BaseModelHasDefault, {"x": 12}),
        (HasDefaultFactory, {"x": [[-2j, 1 + 1j]]}),
        (BaseModelHasDefaultFactory, {"x": [[-2j, 1 + 1j]]}),
    ],
)
def test_pop_sig_with_pydantic_Field(target, kwargs):
    Conf = builds(target, populate_full_signature=True)
    assert instantiate(Conf(**kwargs)) == target(**kwargs)


@pyd_dataclass
class NavbarButton:
    href: AnyUrl


@pyd_dataclass
class Navbar:
    button: NavbarButton


@given(...)
def test_nested_dataclasses(via_yaml: bool):
    navbar = Navbar(button=("https://example.com",))  # type: ignore
    conf = just(navbar)
    if via_yaml:
        # ensure serializable
        assert instantiate(OmegaConf.create(to_yaml(conf))) == navbar
    else:
        assert instantiate(conf) == navbar


class ModelNavbarButton(BaseModel):
    href: AnyUrl


class ModelNavbar(BaseModel):
    button: ModelNavbarButton


@given(...)
def test_nested_base_models(via_yaml: bool):
    navbar = ModelNavbar(button=ModelNavbarButton(href="https://example.com"))  # type: ignore
    conf = just(navbar)
    if via_yaml:
        # ensure serializable
        assert instantiate(OmegaConf.create(to_yaml(conf))) == navbar
    else:
        assert instantiate(conf) == navbar


@pyd_dataclass
class PydFieldNoDefault:
    btwn_0_and_3: int = Field(gt=0, lt=3)


def test_pydantic_Field_no_default():
    Conf = builds(PydFieldNoDefault, populate_full_signature=True)
    out = instantiate(OmegaConf.create(to_yaml(Conf)), btwn_0_and_3=2)
    assert isinstance(out, PydFieldNoDefault) and out == PydFieldNoDefault(
        btwn_0_and_3=2
    )
