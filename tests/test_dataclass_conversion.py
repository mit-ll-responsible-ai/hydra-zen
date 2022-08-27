# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any, Callable

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import OmegaConf
from typing_extensions import TypeAlias

from hydra_zen import builds, instantiate, just, make_config
from hydra_zen.errors import HydraZenUnsupportedPrimitiveError

from .test_just import list_of_objects

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
    field3: Nested = Nested(baz)
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
  _target_: hydra_zen.funcs.get_obj
  path: tests.test_dataclass_conversion.foo
field2:
  _target_: hydra_zen.funcs.get_obj
  path: tests.test_dataclass_conversion.bar
field3:
  _target_: tests.test_dataclass_conversion.Nested
  x:
    _target_: hydra_zen.funcs.get_obj
    path: tests.test_dataclass_conversion.baz
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
def test_make_config_doesnt_convert_dataclasses(dataclass_obj):
    assert make_config(x=dataclass_obj).x is dataclass_obj


@pytest.mark.parametrize("obj", [make_config(), make_config()()])
def test_just_auto_config_raises_on_dynamically_generated_types(obj):
    with pytest.raises(HydraZenUnsupportedPrimitiveError):
        just(obj)
