# Copyright (c) 2026 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import MISSING, InitVar, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Literal, Optional, TypeVar

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import ValidationError
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
from hydra_zen.structured_configs._implementations import (
    _config_rejected_by,
    _resolve_field_default,
    _value_contains_builds,
)
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
class _Inner858:
    x: int = 0


@dataclass
class _OuterDict858:
    outer: "dict[str, _Inner858]"


@dataclass
class _OuterList858:
    outer: "list[_Inner858]"


@dataclass
class _OuterNested858:
    outer: "dict[str, list[_Inner858]]"


@pytest.mark.parametrize(
    "obj",
    [
        _OuterDict858(outer={"a": _Inner858(x=1)}),
        _OuterList858(outer=[_Inner858(x=1)]),
        _OuterNested858(outer={"a": [_Inner858(x=1), _Inner858(x=2)]}),
    ],
)
def test_just_on_container_field_of_typed_dataclasses(obj):
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/858
    # A dataclass whose field is annotated with a container of a dataclass type
    # (e.g. `dict[str, Inner]`) used to fail to convert to yaml because the field's
    # values get converted to `Builds_Inner` configs while the field's annotation
    # retained the original (non-builds) element type, which OmegaConf rejects.
    Conf = just(obj)
    # must not raise omegaconf.errors.ValidationError
    out = instantiate(OmegaConf.create(OmegaConf.to_yaml(Conf)), _convert_="all")
    assert out == obj
    # `==` alone would also hold for a `DictConfig` standing in for `_Inner858`;
    # check that the elements really are instances of the annotated type
    assert all(isinstance(leaf, _Inner858) for leaf in _leaves(out.outer))


def _leaves(x: Any) -> list[Any]:
    if isinstance(x, dict):
        return [leaf for v in x.values() for leaf in _leaves(v)]
    if isinstance(x, (list, tuple)):
        return [leaf for v in x for leaf in _leaves(v)]
    return [x]


def test_value_contains_builds():
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/858
    cfg = just(_Inner858(x=1))
    assert is_builds(cfg)
    assert _value_contains_builds(cfg) is True
    assert _value_contains_builds([0, cfg]) is True
    assert _value_contains_builds({"a": cfg}) is True
    assert _value_contains_builds({"a": [cfg]}) is True
    assert _value_contains_builds([1, 2, 3]) is False
    assert _value_contains_builds({"a": 1}) is False
    assert _value_contains_builds("not a container") is False


@pytest.mark.parametrize(
    "value",
    [
        DictConfig({"a": "???"}),
        DictConfig({"a": "${nope}"}),
        ListConfig(["???"]),
        ListConfig(["${nope}"]),
        {"a": DictConfig({"b": "${nope}"})},
        [ListConfig(["${nope}"])],
    ],
)
def test_value_contains_builds_does_not_resolve_omegaconf_containers(value):
    # Iterating a DictConfig/ListConfig resolves its values; `???` and
    # unresolvable interpolations must survive config creation untouched.
    assert _value_contains_builds(value) is False


_ConfInt858 = builds(int)
_ConfStr858 = builds(str)


class _Plain858:
    # not a dataclass: `_sanitized_type` broadens `list[_Plain858]` to `list[Any]`
    def __init__(self, x: int = 0):
        self.x = x


@pytest.mark.parametrize(
    "type_, value",
    [
        # an unconstrained element type accepts a config
        (list, [just(_Inner858())]),
        (list[Any], [just(_Inner858())]),
        (dict[str, Any], {"a": just(_Inner858())}),
        (Optional[list[Any]], [just(_Inner858())]),
        # ... so does an element type the config is an instance of
        (list[_ConfInt858], [_ConfInt858()]),
        (list[_ConfInt858], [_ConfInt858]),
        (dict[str, _ConfInt858], {"a": _ConfInt858()}),
        (tuple[_ConfInt858, ...], (_ConfInt858(),)),
        # OmegaConf wraps a config assigned to an `Optional[dict]` slot
        (list[Optional[dict]], [just(_Inner858())]),
        # nothing to reject: no config, an OmegaConf container, a kind mismatch
        (list[_Inner858], [1, 2]),
        (list[_Inner858], ListConfig(["${missing}"])),
        (list[_Inner858], {"a": just(_Inner858())}),
        # an element type that is not a class constrains nothing
        (list[Literal[1]], [just(_Inner858())]),
        # a value that is neither a mapping nor a sequence has no elements
        (list[_Inner858], 5),
    ],
)
def test_config_not_rejected_by_annotation(type_, value):
    assert _config_rejected_by(type_, value) is False


@pytest.mark.parametrize(
    "type_, value",
    [
        (list[_Inner858], [just(_Inner858())]),
        (dict[str, _Inner858], {"a": just(_Inner858())}),
        (dict[str, list[_Inner858]], {"a": [just(_Inner858())]}),
        (tuple[_Inner858, ...], (just(_Inner858()),)),
        (Optional[list[_Inner858]], [just(_Inner858())]),
        (list[list[_Inner858]], [[just(_Inner858())]]),
        (list[int], [just(_Inner858())]),
        (list[_ConfInt858], [_ConfStr858()]),
        # a `dict`/`dict[K, V]` or `Optional[list]` slot rejects a config,
        # unlike `Optional[dict]`
        (list[dict], [just(_Inner858())]),
        (list[Optional[list]], [just(_Inner858())]),
    ],
)
def test_config_rejected_by_annotation(type_, value):
    assert _config_rejected_by(type_, value) is True


def test_resolve_field_default():
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/858
    # a mutable default is stored via `default_factory`; resolving it is what
    # lets a nested config be seen
    assert _resolve_field_default(field(default=1)) == 1
    assert _resolve_field_default(field(default_factory=lambda: [1, 2])) == [1, 2]
    assert _resolve_field_default(field()) is MISSING


def _takes_list_858(x: list[int]): ...


def _takes_dict_858(x: dict): ...


def _takes_list_of_any_858(x: list[Any]):
    return x


def _takes_list_of_plain_858(x: list[_Plain858]):
    return x


def _takes_list_of_conf_858(x: list[_ConfInt858]):
    return x


def test_builds_never_calls_a_caller_supplied_default_factory():
    # hydra-zen resolves the factory it wraps a mutable value in, so that a
    # nested config can be seen; a `Field` the caller passed in is left alone,
    # because its factory is user code and must not run during config creation
    calls = []

    def factory():
        calls.append(1)
        return [just(_Inner858())]

    for hydra_recursive in (None, True, False):
        builds(
            _takes_list_858,
            x=field(default_factory=factory),
            hydra_recursive=hydra_recursive,
        )
    assert calls == []


@pytest.mark.parametrize(
    "value",
    [
        DictConfig({"_target_": "${missing}"}),
        DictConfig({"_target_": "???"}),
        ListConfig(["${missing}"]),
    ],
)
def test_caller_supplied_factory_returning_an_omegaconf_container(value):
    # the factory is never called, so its `_target_` node is not resolved: a
    # `???` or an interpolation that cannot resolve yet must not raise here
    Conf = builds(_takes_dict_858, x=field(default_factory=lambda: value))
    assert OmegaConf.to_yaml(Conf)


@pytest.mark.parametrize(
    "target, value, expected_hint",
    [
        # an annotation that already accepts the config keeps its element type,
        # and with it OmegaConf's validation of the container
        (_takes_list_of_any_858, [builds(_Inner858)], list[Any]),
        # `list[_Plain858]` sanitizes to `list[Any]`, which also accepts it
        (_takes_list_of_plain_858, [builds(_Plain858)], list[Any]),
        (_takes_list_of_conf_858, [_ConfInt858()], list[_ConfInt858]),
    ],
)
def test_container_that_accepts_the_config_keeps_its_annotation(
    target, value, expected_hint
):
    Conf = builds(target, x=value)
    assert {f.name: f.type for f in fields(Conf)}["x"] == expected_hint
    assert OmegaConf.to_yaml(Conf)
    with pytest.raises(ValidationError):
        OmegaConf.merge(Conf, {"x": "oops"})


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
    config_maker: Callable[[TDataClass], Builds[type[TDataClass]]],
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
    config_maker: Callable[[list[TDataClass]], list[Builds[type[TDataClass]]]],
    dataclass_obj: TDataClass,
    as_container: Callable[[T], list[T]],
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
