# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import inspect
import sys
from abc import ABC
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    get_type_hints,
)

import hypothesis.strategies as st
import pytest
from hypothesis import given, note, settings
from omegaconf.errors import ValidationError

from hydra_zen import (
    ZenField,
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    mutable_value,
    to_yaml,
)
from hydra_zen.typing import Just
from tests import valid_hydra_literals

Empty = Parameter.empty


def f1(x=2):
    return x


@pytest.mark.parametrize("as_hydrated_dataclass", [False, True])
@given(user_value=valid_hydra_literals, full_signature=st.booleans())
def test_user_specified_value_overrides_default(
    user_value, as_hydrated_dataclass: bool, full_signature: bool
):

    if not as_hydrated_dataclass:
        BuildsF = builds(f1, x=user_value, populate_full_signature=full_signature)
    else:

        @hydrated_dataclass(f1, populate_full_signature=full_signature)
        class BuildsF:
            x: Any = (
                mutable_value(user_value)
                if isinstance(user_value, (list, dict))
                else user_value
            )

    b = BuildsF()
    assert b.x == user_value


def f2(x, y, z, has_default=101):
    return x, y, z, has_default


@settings(max_examples=1000, deadline=None)
@given(
    user_value_x=valid_hydra_literals,
    user_value_y=valid_hydra_literals,
    user_value_z=valid_hydra_literals,
    num_positional=st.integers(0, 3),
    data=st.data(),
)
def test_builds_signature_shuffling_takes_least_path(
    user_value_x,
    user_value_y,
    user_value_z,
    num_positional: int,
    data: st.DataObject,
):
    specified_as_positional = set(["x", "y", "z"][:num_positional])

    if num_positional < 3:
        specified_as_default = data.draw(
            st.lists(st.sampled_from(["x", "y", "z"][num_positional:]), unique=True),
            label="specified_as_default",
        )
    else:
        specified_as_default = data.draw(st.just([]), label="specified_as_default")

    # We will specify an arbitrary selection of x, y, z via `builds`, and then specify the
    # remaining parameters via initializing the resulting dataclass. This ensures that we can
    # accommodate arbitrary "signature shuffling", i.e. that parameters with defaults specified
    # are shuffled just to the right of those without defaults.
    #
    # E.g.
    #  - `builds(f, populate_full_signature=True)`.__init__ -> (x, y, z, has_default=default_value)
    #  - `builds(f, x=1, populate_full_signature=True)`.__init__ -> (y, z, x=1, has_default=default_value)
    #  - `builds(f, y=2, z=-1, populate_full_signature=True)`.__init__ -> (x, y=2, z=-1, has_default=default_value)
    #  - `builds(f, 1, 2, populate_full_signature=True)`.__init__ -> (z=-1, has_default=default_value)

    defaults = dict(x=user_value_x, y=user_value_y, z=user_value_z)

    default_override = {k: defaults[k] for k in specified_as_default}
    positional = [defaults[k] for k in sorted(specified_as_positional)]

    specified_via_init = {
        k: defaults[k]
        for k in set(defaults) - set(specified_as_default) - specified_as_positional
    }

    BuildsF = builds(f2, *positional, **default_override, populate_full_signature=True)
    sig_param_names = [p.name for p in inspect.signature(BuildsF).parameters.values()]
    expected_param_ordering = (
        sorted(specified_via_init) + sorted(specified_as_default) + ["has_default"]
    )
    expected_param_ordering = [
        p for p in expected_param_ordering if p not in specified_as_positional
    ]

    assert sig_param_names == expected_param_ordering

    b = BuildsF(**specified_via_init)
    assert "x" in specified_as_positional and not hasattr(b, "x") or b.x == user_value_x
    assert "y" in specified_as_positional and not hasattr(b, "y") or b.y == user_value_y
    assert "z" in specified_as_positional and not hasattr(b, "z") or b.z == user_value_z
    assert b.has_default == 101

    to_yaml(b)  # should never crash
    x, y, z, has_default = instantiate(b)  # should never crash
    assert x == user_value_x
    assert y == user_value_y
    assert z == user_value_z
    assert has_default == 101


def f3(x: str, *args, y: int = 22, z=[2], **kwargs):
    pass


@pytest.mark.parametrize("include_extra_param", [False, True])
@pytest.mark.parametrize("partial", [None, False, True])
def test_builds_with_full_sig_mirrors_target_sig(
    include_extra_param: bool, partial: Optional[bool]
):

    kwargs = dict(named_param=2) if include_extra_param else {}
    kwargs["y"] = 0  # overwrite default value
    Conf = builds(f3, populate_full_signature=True, zen_partial=partial, **kwargs)

    params = inspect.signature(Conf).parameters.values()

    expected_sig = [("x", str)] if not partial else []

    expected_sig += [("y", int), ("z", Any)]
    if include_extra_param:
        expected_sig.append(("named_param", Any))

    actual_sig = [(p.name, p.annotation) for p in params]
    assert expected_sig == actual_sig

    if not partial:
        conf = Conf(x="-100")
        assert conf.x == "-100"
    else:
        # x should be excluded when partial=True and full-sig is populated
        conf = Conf()  # type: ignore

    assert conf.y == 0
    assert conf.z == [2]

    if include_extra_param:
        assert conf.named_param == 2


def func():
    pass


@dataclass
class ADataClass:
    x: int = 1


def a_func(
    x: int,
    y: str,
    z: bool,
    a_tuple: Tuple[str] = ("hi",),
    optional: Optional[int] = None,
    inferred_optional_str: str = None,
    inferred_optional_any: Mapping = None,
    default: float = 100.0,
    a_function: Callable = func,
    a_class: Type[Dict] = dict,
    a_dataclass: Type[ADataClass] = ADataClass,
):
    pass


class AClass:
    def __init__(
        self,
        x: int,
        y: str,
        z: bool,
        a_tuple: Tuple[str] = ("hi",),
        optional: Optional[int] = None,
        inferred_optional_str: str = None,
        inferred_optional_any: Mapping = None,
        default: float = 100.0,
        a_function: Callable = func,
        a_class: Type[Dict] = dict,
        a_dataclass: Type[ADataClass] = ADataClass,
    ):
        pass

    @classmethod
    def a_class_method(
        cls,
        x: int,
        y: str,
        z: bool,
        a_tuple: Tuple[str] = ("hi",),
        optional: Optional[int] = None,
        inferred_optional_str: str = None,
        inferred_optional_any: Mapping = None,
        default: float = 100.0,
        a_function: Callable = func,
        a_class: Type[Dict] = dict,
        a_dataclass: Type[ADataClass] = ADataClass,
    ):
        pass


class AMetaClass(ABC):
    def __init__(
        self,
        x: int,
        y: str,
        z: bool,
        a_tuple: Tuple[str] = ("hi",),
        optional: Optional[int] = None,
        inferred_optional_str: str = None,
        inferred_optional_any: Mapping = None,
        default: float = 100.0,
        a_function: Callable = func,
        a_class: Type[Dict] = dict,
        a_dataclass: Type[ADataClass] = ADataClass,
    ):
        pass


@pytest.mark.parametrize("target", [a_func, AClass, AClass.a_class_method, AMetaClass])
@given(
    user_specified_values=st.dictionaries(
        keys=st.sampled_from(["x", "y", "z"]), values=st.integers(0, 3), max_size=3
    )
)
def test_builds_partial_with_full_sig_excludes_non_specified_params(
    target, user_specified_values
):
    name_to_type = dict(x=int, y=str, z=bool)
    Conf = builds(
        target,
        **user_specified_values,
        populate_full_signature=True,
        zen_partial=True,
        zen_convert={"dataclass": False},
    )

    expected_sig = [
        (var_name, name_to_type[var_name], user_specified_values[var_name])
        for var_name in sorted(user_specified_values)
    ] + [
        ("a_tuple", Tuple[str], ("hi",)),
        ("optional", Optional[int], None),
        ("inferred_optional_str", Optional[str], None),
        ("inferred_optional_any", Any, None),
        ("default", float, 100.0),
        ("a_function", Any, Conf.a_function),
        ("a_class", Any, Conf.a_class),
        ("a_dataclass", Any, ADataClass),
    ]

    actual_sig = [
        (p.name, p.annotation, p.default)
        for p in inspect.signature(Conf).parameters.values()
    ]
    assert expected_sig == actual_sig

    assert isinstance(Conf.a_function, Just) and "func" in Conf.a_function.path
    assert isinstance(Conf.a_class, Just) and "dict" in Conf.a_class.path


def f_with_fwd_ref(x: "torch.optim.Optimizer"):  # noqa: F821
    return


class A_w_fwd_ref:
    def __init__(self, x: "torch.optim.Optimizer"):  # noqa: F821
        pass


@pytest.mark.parametrize("obj", [f_with_fwd_ref, A_w_fwd_ref])
def test_sig_with_unresolved_fwd_ref(obj):
    # builds should gracefully skip signature parsing for unresolved fwd-references
    instantiate(builds(obj, x=1))


def returns_int() -> int:
    return 1


def expects_int(x: int) -> int:
    return x


@pytest.mark.parametrize(
    "builds_as_default",
    [
        builds(returns_int),  # type
        builds(returns_int)(),  # instance
    ],
)
@pytest.mark.parametrize("hydra_recursive", [True, None])
def test_setting_default_with_builds_widens_type(builds_as_default, hydra_recursive):
    # tests that we address https://github.com/facebookresearch/hydra/issues/1759
    # via auto type-widening
    kwargs = {} if hydra_recursive is None else dict(hydra_recursive=hydra_recursive)
    b = builds(expects_int, x=builds_as_default, **kwargs)
    assert 1 == instantiate(b)  # should not raise ValidationError

    with pytest.raises(ValidationError):
        # ensure that type annotation is broadened only when hydra_recursive=False
        instantiate(builds(expects_int, x=builds_as_default, hydra_recursive=False))


BuildsInt = builds(int)


def f_with_dataclass_annotation(x: BuildsInt = BuildsInt()):
    return x


instantiaters = [
    lambda x: instantiate(
        builds(f_with_dataclass_annotation, populate_full_signature=True), x=x
    ),
    lambda x: instantiate(builds(f_with_dataclass_annotation, x=x)),
]


@pytest.mark.parametrize(
    "bad_value",
    [
        1,  # not a structured config
        builds(str)(),  # instance of different structured config
    ],
)
@pytest.mark.parametrize("instantiater", instantiaters)
def test_builds_doesnt_widen_dataclass_type_annotation(bad_value, instantiater):
    with pytest.raises(ValidationError):
        instantiater(bad_value)


@pytest.mark.parametrize("instantiater", instantiaters)
def test_dataclass_type_annotation_with_subclass_default(instantiater):
    # ensures that configs that inherite from a base class used
    # in an annotation passes Hydra's validation
    Child = builds(str, builds_bases=(BuildsInt,))
    assert instantiater(Child()) == ""


class LooksLikeBuilds:
    _target_: str = "hello world"


def f_with_builds_like_annotation(x: LooksLikeBuilds):
    pass


def test_builds_widens_non_dataclass_type_with_target():
    Conf = builds(f_with_builds_like_annotation, x=builds(int))
    hints = get_type_hints(Conf)
    assert hints["x"] is Any


def func_with_list_annotation(x: List[int]):
    return x


def test_type_widening_with_internal_conversion_to_Builds():
    # This test relies on omegaconf <= 2.1.1 to exercise the desired behavior,
    # but should pass regardless.
    #
    # We contrive a case where an annotation is supported by Hydra, but the supplied
    # value requires us to cast internally to a targeted conf; this is due to the
    # downstream patch: https://github.com/mit-ll-responsible-ai/hydra-zen/pull/172
    #
    # Thus in this case we detect that, even though the original configured value is
    # valid, we must represent the value with a structured config. Thus we should widen
    # the annotation from `List[int]` to `Any`.
    #
    # Generally, in cases where we need to internally cast to a Builds, the associated
    # type annotation is not supported by Hydra to begin with, and thus is broadened
    # via type-sanitization.
    Base = make_config(x=1)
    Conf = builds(func_with_list_annotation, x=[1, 2], builds_bases=(Base,))
    instantiate(Conf)  # shouldn't raise


def test_type_widening_for_interpolated_field_is_needed():
    @dataclass
    class Config:
        x: str = "${A}"

    with pytest.raises(ValidationError):
        instantiate(Config, x=builds(str))


def test_type_widening_for_interpolated_field():
    C1 = make_config(x=ZenField(str, "A"))
    assert get_type_hints(C1)["x"] is str

    # field is interpolated entry
    C2 = make_config(x=ZenField(str, "${A}"))
    assert get_type_hints(C2)["x"] is Any


def use_data(data: List[float]):
    return data


def get_data():
    return [5.0, 2.0]


def test_type_widening_for_interpolated_field_regression_example():
    cfg = builds(use_data, data="${data}")
    Config = make_config(data=builds(get_data), out=cfg)
    obj = instantiate(Config)  # shouldn't raise
    assert obj.out == get_data()


def func_with_various_defaults(x=1, y="a", z=[1, 2]):
    pass


valid_defaults = (
    st.none()
    | st.booleans()
    | st.text(alphabet="abcde")
    | st.integers(-3, 3)
    | st.lists(st.integers(-2, 2))
    | st.fixed_dictionaries({"a": st.integers(-2, 2)})
)


@given(
    passed_args=st.fixed_dictionaries(
        {}, optional=dict(x=st.just(1), y=st.just("a"), z=st.just([1, 2]))
    ),
    parent=st.just(())
    | st.dictionaries(st.sampled_from(["x", "y", "z"]), valid_defaults).map(
        lambda kw: (make_config(**kw),)
    ),
)
def test_pop_full_sig_is_always_identical_to_manually_specifying_sig_args(
    passed_args: dict, parent: tuple
):
    Conf = builds(
        func_with_various_defaults,
        x=1,
        y="a",
        z=[1, 2],
        populate_full_signature=False,
        builds_bases=parent,
    )
    ConfWithPopSig = builds(
        func_with_various_defaults,
        **passed_args,
        builds_bases=parent,
        populate_full_signature=True,
    )
    if parent:
        note(str(parent[0]()))
    note(to_yaml(Conf))
    note(to_yaml(ConfWithPopSig))
    assert to_yaml(Conf) == to_yaml(ConfWithPopSig)


class Color(Enum):
    red = 1
    green = 2


def f_with_color(x: Color = Color.red):
    return x


def test_populate_annotated_enum_regression():
    assert instantiate(builds(f_with_color, populate_full_signature=True)) is Color.red


class A:
    # Manually verified using `inspect.signature` after
    # the fix of https://bugs.python.org/issue40897
    py_310_sig = (("a", int),)

    def __new__(cls, a: int):
        return object.__new__(cls)


class B(A):
    py_310_sig = (("b", float),)

    def __init__(self, b: float):
        pass


class C(A):
    py_310_sig = (("c", str),)

    def __new__(cls, c: str):
        return object.__new__(cls)


class D(A):
    py_310_sig = (("a", int),)


class E(B):
    py_310_sig = (("a", int),)


@pytest.mark.parametrize("Obj", [A, B, C, D, E])
def test_parse_sig_with_new_vs_init(Obj):
    # Ensure that `builds` inspects __new__ for signature and annotations
    # with same priority as `inspect.signature` in Python >= 3.9.1
    Conf = builds(Obj, populate_full_signature=True)

    sig_via_builds = tuple(
        (p.name, p.annotation) for p in inspect.signature(Conf).parameters.values()
    )

    assert sig_via_builds == Obj.py_310_sig


def test_Counter():
    # Counter has an interesting signature
    # Python 3.7: (*args, **kwds)                   -- no self!
    #      >=3.8: (self, iterable=None, /, **kwds)  -- pos-only
    assert instantiate(builds(Counter, [1, 1, 2, 1])) == Counter([1, 1, 2, 1])
    assert instantiate(builds(Counter, a=1, b=2)) == Counter(a=1, b=2)

    if sys.version_info > (3, 8):
        with pytest.raises(TypeError):
            # signature: Counter(iterable=None, /, **kwds)
            builds(Counter, [1], [2])


def f_x(x: int):
    return int


def test_inheritance_populates_init_field():
    Parent = make_config(x=1)
    Conf1 = builds(f_x, populate_full_signature=True)
    Conf2 = builds(f_x, populate_full_signature=True, builds_bases=(Parent,))

    with pytest.raises(TypeError):
        # x is missing
        Conf1()  # type: ignore

    # x is 'filled' via inheritance
    Conf2()


class A265:
    @classmethod
    def foo(cls):
        return cls.bar()

    @classmethod
    def bar(cls):
        raise NotImplementedError()


class B265(A265):
    @classmethod
    def bar(cls):
        return 1


class C265(B265):
    @classmethod
    def bar(cls):
        return "hello"


@pytest.mark.parametrize("Cls", [B265, C265])
def test_builds_of_inherited_classmethod(Cls):
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/265

    assert instantiate(builds(Cls.foo)) == Cls.foo()
