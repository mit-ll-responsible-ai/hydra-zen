# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import MISSING, InitVar, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Type, TypeVar

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf
from typing_extensions import TypeAlias

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    just,
    make_config,
    make_custom_builds_fn,
    mutable_value,
)
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError
from hydra_zen.structured_configs._type_guards import is_builds
from hydra_zen.typing import Builds
from hydra_zen.typing._implementations import DataClass_, InstOrType, ZenConvert

from .test_just import list_of_objects

TDataClass = InstOrType[DataClass_]
T = TypeVar("T")
Interface1: TypeAlias = Callable[[int], int]
Interface2: TypeAlias = Callable[[str], str]


@dataclass
class Nested:
    x: Any


def foo(i: int) -> int:
    return i


def bar(s: str) -> str:
    return s


def baz(f: float) -> float:
    return f


@dataclass
class Stuff:
    field1: Interface1
    field2: Interface2
    field3: Nested = mutable_value(Nested(baz))
    field4: Any = field(default_factory=lambda: [1, 2, 3])


@pytest.mark.parametrize("via_just", [True, False])
def test_293_proposal(via_just: bool):
    # https://github.com/mit-ll-responsible-ai/hydra-zen/discussions/293
    foobar = Stuff(foo, bar, Nested(baz))
    BuildsFooBar2 = (
        just(foobar)
        if via_just
        else builds(Stuff, field1=foo, field2=bar, populate_full_signature=True)
    )
    inst = instantiate(BuildsFooBar2)
    assert isinstance(inst, Stuff)
    assert inst == foobar
    assert (
        OmegaConf.to_yaml(BuildsFooBar2)
        == """\
_target_: tests.test_dataclass_conversion.Stuff
field1:
  path: tests.test_dataclass_conversion.foo
  _target_: hydra_zen.funcs.get_obj
field2:
  path: tests.test_dataclass_conversion.bar
  _target_: hydra_zen.funcs.get_obj
field3:
  _target_: tests.test_dataclass_conversion.Nested
  x:
    path: tests.test_dataclass_conversion.baz
    _target_: hydra_zen.funcs.get_obj
field4:
- 1
- 2
- 3
"""
    )


list_of_objects = [
    1,
    None,
    True,
    1 + 2j,
    "hi",
    b"hi",
    dict(a=1),
    [1, 2],
    Path().home(),
    Stuff,
    Stuff(foo, bar, Nested(baz)),
]


def test_recursive_manual():
    Conf = just(Nested(Nested(Nested)))
    assert just(Conf) is Conf
    out = instantiate(Conf)
    assert isinstance(out, Nested)
    assert isinstance(out.x, Nested)
    assert out.x.x is Nested


@given(
    st.recursive(
        st.sampled_from(list_of_objects), lambda x: st.builds(Nested, x), max_leaves=5
    )
)
def test_recursive_conversion(x):
    if not isinstance(x, Nested):
        x = Nested(x)
    assert instantiate(just(x)) == x
    assert instantiate(builds(Nested, x=x.x)) == x
    assert instantiate(OmegaConf.create(OmegaConf.to_yaml(just(x)))) == x


@dataclass
class NoDefault:
    x: Any


@dataclass
class HasDefault:
    x: Any = 1


@dataclass
class HasDefaultFactory:
    x: Any = field(default_factory=lambda: [1 + 2j, "a", 3])


@dataclass
class HasNonInit:
    x: float
    y: float = field(init=False)

    def __post_init__(self):
        self.y = self.x * 2


@dataclass
class HasInitVarNoDefault:
    x: InitVar[Any]


@dataclass
class HasInitVarWithDefault:
    x: InitVar[Any] = 22


@pytest.mark.parametrize(
    "dataclass_inst",
    [
        NoDefault(x=1),
        HasDefault(),
        HasDefaultFactory(),
        HasDefaultFactory([Path.home(), 2 - 4j]),
        HasNonInit(x=22),
    ],
)
def test_flavors_of_dataclasses(dataclass_inst):
    Config = just(dataclass_inst)
    inst = instantiate(Config)
    assert isinstance(inst, type(dataclass_inst))
    assert inst.x == dataclass_inst.x


@pytest.mark.parametrize(
    "dataclass_inst",
    [
        HasInitVarNoDefault(x=1),
        HasInitVarWithDefault(),
    ],
)
def test_just_raises_on_initvar_field(dataclass_inst):
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(dataclass_inst)


@pytest.mark.parametrize(
    "dataclass_type",
    [
        HasInitVarNoDefault,
        HasInitVarWithDefault,
    ],
)
def test_builds_with_initvar_field(dataclass_type):
    assert instantiate(
        builds(dataclass_type, populate_full_signature=True), x=1
    ) == dataclass_type(x=1)


def _zen_get(obj):
    x = obj.default.default
    return x if x is not MISSING else obj.default.default_factory()


@pytest.mark.parametrize(
    "dataclass_obj",
    [
        NoDefault(x=1),
        HasDefault(),
        HasDefaultFactory(),
        HasDefaultFactory([Path.home(), 2 - 4j]),
        HasNonInit(x=22),
        builds(int)(),
        NoDefault,
        HasDefault,
        HasDefaultFactory,
        HasDefaultFactory,
        HasNonInit,
        builds(int),
    ],
)
@pytest.mark.parametrize(
    "config_maker",
    [
        lambda x: make_config(x=x)().x,
        lambda x: builds(dict, x=x, zen_convert={"dataclass": False})().x,
        lambda x: make_custom_builds_fn(zen_convert={"dataclass": False})(
            dict, x=x
        )().x,
        lambda x: just(x, zen_convert={"dataclass": False}),
        lambda x: _zen_get(
            ZenField(default=x, name="x", zen_convert={"dataclass": False})
        ),
    ],
)
def test_no_dataclass_conversion(
    config_maker: Callable[[TDataClass], TDataClass], dataclass_obj: TDataClass
):
    assert config_maker(dataclass_obj) is dataclass_obj


@pytest.mark.parametrize(
    "dataclass_obj",
    [
        NoDefault(x=1),
        HasDefault(),
        HasDefaultFactory(),
        HasDefaultFactory([Path.home(), 2 - 4j]),
        HasNonInit(x=22),
        NoDefault,
        HasDefault,
        HasDefaultFactory,
        HasDefaultFactory,
        HasNonInit,
    ],
)
@pytest.mark.parametrize(
    "config_maker",
    [
        lambda x: make_config(x=x, zen_convert={"dataclass": True})().x,
        lambda x: make_custom_builds_fn(zen_convert={"dataclass": True})(dict, x=x)().x,
        lambda x: builds(dict, x=x)().x,
        lambda x: just(x),
        lambda x: _zen_get(
            ZenField(
                default=x,
                name="x",
            )
        ),
    ],
)
def test_yes_dataclass_conversion(
    config_maker: Callable[[TDataClass], Builds[Type[TDataClass]]],
    dataclass_obj: TDataClass,
):
    out = config_maker(dataclass_obj)
    inst_out = instantiate(out)
    assert out is not dataclass_obj
    assert is_builds(out)
    assert isinstance(inst_out, type(dataclass_obj))
    assert inst_out == dataclass_obj


def identity(x):
    return x


def test_builds_with_positional_arg():
    out1 = instantiate(
        builds(
            identity,
            HasDefault(),
            hydra_convert="all",
            zen_convert={"dataclass": True},
        )
    )
    assert isinstance(out1, HasDefault) and out1 == HasDefault()

    out2 = instantiate(
        builds(
            identity,
            HasDefault(),
            hydra_convert="all",
            zen_convert={"dataclass": False},
        )
    )
    assert not isinstance(out2, HasDefault)


@pytest.mark.parametrize(
    "dataclass_obj",
    [
        NoDefault(x=1),
        HasDefault(),
        HasDefaultFactory(),
        HasDefaultFactory([Path.home(), 2 - 4j]),
        HasNonInit(x=22),
        NoDefault,
        HasDefault,
        HasDefaultFactory,
        HasDefaultFactory,
        HasNonInit,
    ],
)
@pytest.mark.parametrize(
    "config_maker",
    [
        lambda x: make_config(x=x, zen_convert={"dataclass": True})().x,
        lambda x: builds(dict, x=x)().x,
        lambda x: make_custom_builds_fn()(dict, x=x)().x,
        lambda x: just(x),
        lambda x: ZenField(
            default=x,
            name="x",
        ).default.default_factory(),  # type: ignore
    ],
)
@pytest.mark.parametrize(
    "as_container",
    [
        lambda dataclass_obj: [0, dataclass_obj],
        lambda dataclass_obj: {1: dataclass_obj},
    ],
)
def test_recursive_dataclass_conversion(
    config_maker: Callable[[List[TDataClass]], List[Builds[Type[TDataClass]]]],
    dataclass_obj: TDataClass,
    as_container: Callable[[T], List[T]],
):
    out = config_maker(as_container(dataclass_obj))[1]  # type: ignore
    inst_out = instantiate(out)
    assert out is not dataclass_obj
    assert is_builds(out)
    assert isinstance(inst_out, type(dataclass_obj))
    assert inst_out == dataclass_obj


@pytest.mark.parametrize("obj", [make_config(), make_config()()])
def test_just_auto_config_raises_on_dynamically_generated_types(obj):
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(obj)


@dataclass
class A:
    y: int = 1


A_dictconfig = instantiate(A)


@pytest.mark.parametrize(
    "config, expected",
    [
        (
            builds(dict, x=A()),
            dict(x=A()),
        ),
        (
            builds(dict, x=A(), zen_convert=ZenConvert(dataclass=True)),
            dict(x=A()),
        ),
        (
            builds(dict, x=A(), zen_convert=ZenConvert(dataclass=False)),
            dict(x=A_dictconfig),
        ),
        (
            make_config(
                x=A(), zen_convert=ZenConvert(dataclass=True), hydra_convert="all"
            ),
            dict(x=A()),
        ),
        (
            make_config(
                x=A(),
                zen_convert=ZenConvert(dataclass=False),
            ),
            dict(x=A_dictconfig),
        ),
    ],
)
def test_zen_convert(config, expected):
    actual = instantiate(config)
    assert actual == expected
    assert isinstance(actual["x"], type(expected["x"]))


def f(x=HasDefault()):
    return x


def test_hydrated_dataclass():
    @hydrated_dataclass(f, populate_full_signature=True)
    class A: ...

    @hydrated_dataclass(
        f, populate_full_signature=True, zen_convert={"dataclass": False}
    )
    class B: ...

    out_a = instantiate(A)
    assert isinstance(out_a, HasDefault)
    assert out_a == HasDefault()

    out_b = instantiate(B)
    assert not isinstance(out_b, HasDefault)
    assert out_b == HasDefault()
