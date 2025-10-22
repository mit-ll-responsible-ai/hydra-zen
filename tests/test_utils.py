# Copyright (c) 2025 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import collections.abc as abc
import enum
import functools
import pickle
import random
import string
import sys
from dataclasses import InitVar, dataclass, field as dataclass_field, make_dataclass
from inspect import signature
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    Annotated,
    Any,
    Callable,
    Dict,
    Final,
    List,
    NewType,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given, settings
from omegaconf import II, OmegaConf, ValidationError
from omegaconf.errors import (
    ConfigIndexError,
    ConfigTypeError,
    ConfigValueError,
    KeyValidationError,
)
from typing_extensions import (
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeVarTuple,
    Unpack,
)

from hydra_zen import DefaultBuilds, builds, instantiate, mutable_value
from hydra_zen._compatibility import HYDRA_VERSION, Version, _get_version
from hydra_zen.funcs import partial_with_wrapper
from hydra_zen.structured_configs._utils import (
    StrictDataclassOptions,
    field,
    is_interpolated_string,
    merge_settings,
    safe_name,
)
from hydra_zen.typing import Builds
from hydra_zen.typing._implementations import AllConvert, ZenConvert
from tests import everything_except

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")


def pass_through(*args):
    return args


def pass_through_kwargs(**kwargs):
    return kwargs


class C:
    def __repr__(self):
        return "C as a repr"

    def f(self):
        return


def f():
    pass


@pytest.mark.parametrize(
    "obj, expected_name",
    [
        (1, "1"),
        (dict, "dict"),
        (C, "C"),
        (C(), "C as a repr"),
        ("moo", "'moo'"),
        (None, "None"),
        (f, "f"),
    ],
)
def test_safename_known(obj, expected_name):
    assert safe_name(obj) == expected_name


@given(
    st.from_type(type)
)  # this draws any type that has a strategy registered with hypothesis!
def test_fuzz_safename(obj):
    safe_name(obj)


def test_mutable_values():
    @dataclass
    class A:
        a_list: list[int] = mutable_value([1, 2, 3])
        a_dict: dict[str, int] = mutable_value(dict(a=1))

    a = A()
    assert a.a_dict == {"a": 1}
    assert a.a_list == [1, 2, 3]


def test_documented_instantiate_example():
    from hydra_zen import builds, instantiate

    assert instantiate(builds(dict, a=1, b=2), c=3) == dict(a=1, b=2, c=3)
    assert instantiate(builds(list, (1, 2, 3))) == [1, 2, 3]


class Color(enum.Enum):
    pass


class SomeProtocol(Protocol[T]):  # type: ignore
    ...


NoneType: TypeAlias = None


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),  # supported primitives
        (float, float),
        (str, str),
        (bool, bool),
        (bytes, bytes),
        (Path, Path),
        (PosixPath, Path),
        (WindowsPath, Path),
        (Color, Color),
        (C, Any),  # unsupported primitives
        (type(None), Any),
        (set, Any),
        (list, list),
        (tuple, tuple),
        (dict, dict),
        (callable, Any),
        (frozenset, Any),
        (List, list),
        (Dict, dict),
        (T, Any),
        (List[T], list[Any]),
        (Tuple[T, T], tuple[Any, Any]),
        (Callable[P, int], Any),
        (P, Any),
        (P.args, Any),  # type: ignore
        (P.kwargs, Any),  # type: ignore
        (Ts, Any),
        (SomeProtocol[T], Any),
        (tuple[Unpack[Ts]], tuple[Any, ...]),
        (tuple[Unpack[Ts], int], tuple[Any, ...]),
        pytest.param(
            tuple[str, Unpack[Ts]],
            tuple[Any, ...],
            marks=[
                pytest.mark.xfail(
                    HYDRA_VERSION < Version(1, 2, 0),
                    reason="Hydra 1.1.2 doesn't parse tuples beyond first element.",
                ),
            ],
        ),
        pytest.param(
            tuple[str, Unpack[Ts], int],
            tuple[Any, ...],
            marks=[
                pytest.mark.xfail(
                    HYDRA_VERSION < Version(1, 2, 0),
                    reason="Hydra 1.1.2 doesn't parse tuples beyond first element.",
                ),
            ],
        ),
        (Annotated[int, int], int),
        (Annotated[tuple[str, str], int], tuple[str, str]),
        (Annotated[Builds, int], Any),
        (NewType("I", int), int),
        (NewType("S", tuple[str, str]), tuple[str, str]),
        (Self, Any),  # type: ignore
        (Literal[1, 2], Any),  # unsupported generics
        (type[int], Any),
        (Builds, Any),
        (Builds[int], Any),
        (type[Builds[int]], Any),
        (set, Any),
        (set[int], Any),
        (Final[int], Any),
        (Callable, Any),
        (Callable[[int], int], Any),
        (abc.Callable, Any),
        (abc.Mapping, Any),
        (Union[str, int], Any),
        (Optional[frozenset], Any),
        (Union[NoneType, frozenset], Any),
        (Union[NoneType, int], Optional[int]),  # supported Optional
        (Optional[Color], Optional[Color]),
        (Optional[list[Color]], Optional[list[Color]]),
        (
            Optional[list[list[int]]],
            Optional[list[list[int]]],
        ),
        (list[int], list[int]),  # supported containers
        (list[frozenset], list[Any]),
        (
            list[list[int]],
            list[list[int]],
        ),
        (list[tuple[int, int]], list[Any]),
        (list[T], list[Any]),
        (dict[str, float], dict[str, float]),
        (dict[C, int], dict[Any, int]),
        (dict[str, C], dict[str, Any]),
        (dict[C, C], dict[Any, Any]),
        (
            dict[str, list[int]],
            dict[str, list[int]],
        ),
        (tuple[str], tuple[str]),
        (tuple[str, ...], tuple[str, ...]),
        (tuple[str, str, str], tuple[str, str, str]),
        (
            tuple[list[int]],
            (tuple[list[int]]),
        ),
        (Union[NoneType, tuple[int, int]], Optional[tuple[int, int]]),
        (Union[tuple[int, int], NoneType], Optional[tuple[int, int]]),
        (
            list[dict[str, list[int]]],
            list[dict[str, list[int]]],
        ),
        (
            list[list[type[int]]],
            list[list[Any]],
        ),
        (tuple[tuple[int, ...], ...], tuple[Any, ...]),
        (Optional[tuple[tuple[int, ...], ...]], Optional[tuple[Any, ...]]),
        (InitVar[list[frozenset]], Any if sys.version_info < (3, 8) else list[Any]),
    ],
)
def test_sanitized_type_expected_behavior(in_type, expected_type):
    assert DefaultBuilds._sanitized_type(in_type) == expected_type, in_type

    if in_type != expected_type and (in_type, expected_type) not in [
        (List, list),
        (Dict, dict),
    ]:
        # In cases where we change the type, it should be because omegaconf
        # doesn't support that annotation.
        # This check will help catch cases where omegaconf/hydra has added support for
        # new annotations like Literal
        @dataclass
        class Bad:
            x: in_type

        with pytest.raises(
            (
                ValidationError,
                AssertionError,
                ConfigTypeError,
                KeyValidationError,
                ConfigIndexError,
                ConfigValueError,
                AttributeError,
            )
        ):
            Conf = OmegaConf.create(Bad)  # type: ignore
            Conf.x = [[int]]  # special case: validate

    @dataclass
    class Tmp:
        x: expected_type

    # shouldn't raise on `expected_type`
    OmegaConf.create(Tmp)  # type: ignore


def test_tuple_annotation_normalization():
    x = DefaultBuilds._sanitized_type(Tuple[int, str, int])
    assert tuple[Any, Any, Any] == tuple[Any, Any, Any], "yee"
    assert (
        DefaultBuilds._sanitized_type(Tuple[int, str, int]) == tuple[Any, Any, Any]
    ), x


def f_list(x: List):
    return x


def f_dict(x: Dict):
    return x


def f_tuple(x: Tuple):
    return x


@pytest.mark.parametrize(
    "func, value", [(f_list, [1]), (f_dict, dict(a=1)), (f_tuple, (1,))]
)
def test_bare_generics(func, value):
    # python is super flaky about how it represents bare generics.. so
    # we have to write a separate test for them here where we don't
    # check the output of sanitize_type
    assert instantiate(builds(func, populate_full_signature=True, x=value)) == value


def test_vendored_field():
    # Test that our implementation of `field` matches that of `dataclasses.field

    # The case where `default` is specified instead of `default_factory`
    # is already covered via our other tests

    our_field = field(default_factory=lambda: list([1, 2]))
    their_field = dataclass_field(default_factory=lambda: list([1, 2]))

    @dataclass
    class A:
        x: Any = our_field

    @dataclass
    class B:
        x: Any = their_field

    assert isinstance(our_field, type(their_field))
    assert hasattr(A, "x") is hasattr(B, "x")
    assert A().x == B().x


def test_builds_random_regression():
    # was broken in `0.3.0rc3`
    assert 1 <= instantiate(builds(random.uniform, 1, 2)) <= 2


def f_for_interp(*args, **kwargs):
    return args[0]


# II renders a string in omegaconf's interpolated-field format
@given(st.text(alphabet=string.ascii_lowercase, min_size=1).map(II))
def test_is_interpolated_against_omegaconf_generated_interpolated_strs(text):
    assert is_interpolated_string(text)
    assume(text != "config")
    # ensure interpolation actually works
    assert instantiate(builds(f_for_interp, text), **{text[2:-1]: 1}) == 1


@settings(deadline=None)
@given(everything_except(str))
def test_non_strings_are_not_interpolated_strings(not_a_str):
    assert not is_interpolated_string(not_a_str)


@given(st.text(alphabet=string.printable))
def test_strings_that_fail_to_interpolate_are_not_interpolated_strings(any_text):
    c = builds(
        f_for_interp, any_text
    )  # any_text is an attempt at an interpolated field
    kwargs = {any_text[2:-1]: 1}
    try:
        # Interpreter raises if `any_text` is not a valid field name
        # omegaconf raises if `any_text` causes a grammar error
        out = instantiate(c, **kwargs)
    except Exception:
        # either fail case means `any_text` is not a valid interpolated string
        assert not is_interpolated_string(any_text)
        return

    # If `any_text` is a valid interpolated string, then `out == 1`
    assert out == 1 or not is_interpolated_string(any_text)


@given(
    major=st.integers(0, 100),
    minor=st.integers(0, 100),
    patch=st.integers(0, 100),
    # tests Hydra-style and OmegaConf-style dev/rc-style
    # release strings. See:
    # https://pypi.org/project/hydra-core/#history
    # https://pypi.org/project/omegaconf/#history
    patch_suffix=st.just("")
    | st.integers(0, 100).map(lambda x: f"rc{x}")
    | st.integers(0, 100).map(lambda x: f".dev{x}"),
)
def test_get_version(major: int, minor: int, patch: int, patch_suffix: str):
    version_string = f"{major}.{minor}.{patch}{patch_suffix}"
    expected = Version(major, minor, patch)
    assert _get_version(version_string) == expected


def test_version_comparisons():
    assert Version(1, 0, 0) == Version(1, 0, 0)
    assert Version(1, 0, 0) <= Version(1, 0, 0)
    assert Version(1, 0, 0) != Version(2, 0, 0)
    assert Version(1, 1, 1) < Version(2, 0, 0)
    assert Version(1, 0, 0) < Version(2, 0, 0)
    assert Version(1, 0, 0) < Version(1, 1, 0)
    assert Version(1, 0, 0) < Version(1, 0, 1)
    assert Version(1, 0, 1) < Version(1, 1, 0)


@given(...)
def test_merge_settings_idempotence(
    user_settings: Optional[ZenConvert], default_settings: AllConvert
):
    merged_1 = merge_settings(user_settings, default_settings)
    merged_2 = merge_settings(merged_1, default_settings)  # type: ignore
    assert merged_1 == merged_2


@given(...)
def test_merge_settings_retains_user_settings(
    user_settings: Optional[ZenConvert], default_settings: AllConvert
):
    merged = merge_settings(user_settings, default_settings)
    assert default_settings
    if user_settings is None:
        user_settings = {}
    for k, v in merged.items():
        if k in user_settings:
            assert user_settings[k] == v
        else:
            assert default_settings[k] == v

    # test inputs are not mutated
    merged["apple"] = 22  # type: ignore
    assert "apple" not in user_settings
    assert "apple" not in default_settings


@pytest.mark.parametrize("bad_settings", [1, {"not a field": True}, {"dataclass": 1.0}])
def test_merge_settings_validation(bad_settings):
    with pytest.raises((TypeError, ValueError)):
        merge_settings(bad_settings, {"dataclass": True, "flat_target": True})


def test_strict_dataclass_options_reflects_current_dataclass_ver():
    strict_keys = set(StrictDataclassOptions.__required_keys__) | set(
        StrictDataclassOptions.__optional_keys__
    )
    actual_keys = set(signature(make_dataclass).parameters)
    actual_keys.remove("fields")
    # Python 3.14 adds a `decorator` parameter that we intentionally exclude
    actual_keys.discard("decorator")
    assert strict_keys == actual_keys


def pfunc(x: int, y: int = 1):
    return x + y


def pwrapper(func):
    if hasattr(func, "__wrapped__"):
        func.__wrapped__ += 1
    else:
        func.__wrapped__ = 1
    return func


@pytest.mark.parametrize("use_pickle", [True, False], ids=["pickle", "no_pickle"])
def test_partial_with_wrapper(use_pickle: bool):
    if hasattr(pfunc, "__wrapped__"):
        del pfunc.__wrapped__  # type: ignore
    p = partial_with_wrapper((pwrapper,), pfunc, 1, y=2)
    if use_pickle:
        p = pickle.loads(pickle.dumps(p))
    assert isinstance(p, functools.partial)
    assert not hasattr(pfunc, "__wrapped__")
    assert p() == 3
    assert pfunc.__wrapped__ == 1  # type: ignore

    p2 = partial_with_wrapper((pwrapper,), p, y=3)
    if use_pickle:
        p2 = pickle.loads(pickle.dumps(p2))
    assert isinstance(p2, partial_with_wrapper)
    assert pfunc.__wrapped__ == 1  # type: ignore
    # wrapper should be applied twice
    assert p2() == 4
    assert p2.func is pfunc
    assert pfunc.__wrapped__ == 3  # type: ignore


@given(...)
def test_partial_parity(
    args1: List[int], args2: List[int], kwargs1: Dict[str, int], kwargs2: Dict[str, int]
):
    def f(*args, **kwargs):
        return args, kwargs

    pw = partial_with_wrapper((pwrapper,), f, *args1, **kwargs1)
    p = functools.partial(f, *args1, **kwargs1)
    assert isinstance(pw, partial_with_wrapper)
    assert isinstance(p, functools.partial)
    assert pw.func is p.func
    assert pw.args == p.args
    assert pw.keywords == p.keywords

    pw2 = partial_with_wrapper((pwrapper,), pw, *args2, **kwargs2)
    p2 = functools.partial(p, *args2, **kwargs2)
    assert isinstance(pw2, partial_with_wrapper)
    assert isinstance(p2, functools.partial)
    assert pw2.func is p2.func
    assert pw2.args == p2.args
    assert pw2.keywords == p2.keywords
