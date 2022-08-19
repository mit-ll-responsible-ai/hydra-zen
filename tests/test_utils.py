# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import collections.abc as abc
import enum
import random
import string
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path, PosixPath, WindowsPath
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NewType,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from omegaconf import II, OmegaConf, ValidationError
from omegaconf.errors import (
    ConfigIndexError,
    ConfigTypeError,
    ConfigValueError,
    KeyValidationError,
)
from typing_extensions import (
    Annotated,
    Final,
    Literal,
    ParamSpec,
    Protocol,
    Self,
    TypeAlias,
    TypeVarTuple,
    Unpack,
)

from hydra_zen import builds, instantiate, mutable_value
from hydra_zen._compatibility import (
    HYDRA_SUPPORTS_BYTES,
    HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES,
    HYDRA_VERSION,
    OMEGACONF_VERSION,
    HYDRA_SUPPORTS_Path,
    Version,
    _get_version,
)
from hydra_zen.structured_configs._utils import (
    field,
    is_interpolated_string,
    safe_name,
    sanitized_type,
)
from hydra_zen.typing import Builds
from tests import everything_except

T = TypeVar("T")
P = ParamSpec("P")
Ts = TypeVarTuple("Ts")

IS_PY37 = (3, 6) < sys.version_info < (3, 8)


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
        a_list: List[int] = mutable_value([1, 2, 3])
        a_dict: Dict[str, int] = mutable_value(dict(a=1))

    a = A()
    assert a.a_dict == {"a": 1}
    assert a.a_list == [1, 2, 3]


def test_documented_instantiate_example():
    from hydra_zen import builds, instantiate

    assert instantiate(builds(dict, a=1, b=2), c=3) == dict(a=1, b=2, c=3)
    assert instantiate(builds(list, (1, 2, 3))) == [1, 2, 3]


class Color(enum.Enum):
    pass


class SomeProtocol(Protocol[T]):
    ...


NoneType: TypeAlias = None


@pytest.mark.parametrize(
    "in_type, expected_type",
    [
        (int, int),  # supported primitives
        (float, float),
        (str, str),
        (bool, bool),
        (bytes, bytes if HYDRA_SUPPORTS_BYTES else Any),
        (Path, Path if HYDRA_SUPPORTS_Path else Any),
        (PosixPath, Path if HYDRA_SUPPORTS_Path else Any),
        (WindowsPath, Path if HYDRA_SUPPORTS_Path else Any),
        (Color, Color),
        (C, Any),  # unsupported primitives
        (type(None), Any),
        (set, Any),
        (list, (List if OMEGACONF_VERSION < Version(2, 2, 3) else list)),
        (tuple, (Tuple if OMEGACONF_VERSION < Version(2, 2, 3) else tuple)),
        (dict, (Dict if OMEGACONF_VERSION < Version(2, 2, 3) else dict)),
        (callable, Any),
        (frozenset, Any),
        (List, List if not IS_PY37 else List[Any]),
        (Tuple, Tuple),
        (Dict, Dict if not IS_PY37 else Dict[Any, Any]),
        (T, Any),
        (List[T], List[Any]),
        (Tuple[T, T], Tuple[Any, Any]),
        (Callable[P, int], Any),
        (P, Any),
        (P.args, Any),  # type: ignore
        (P.kwargs, Any),  # type: ignore
        (Ts, Any),
        (SomeProtocol[T], Any),
        pytest.param(
            Tuple[Unpack[Ts]],
            Tuple[Any, ...],
            marks=pytest.mark.skipif(
                sys.version_info <= (3, 7), reason="Python 3.6 doesn't support Unpack"
            ),
        ),
        pytest.param(
            Tuple[Unpack[Ts], int],
            Tuple[Any, ...],
            marks=pytest.mark.skipif(
                sys.version_info < (3, 7), reason="Python 3.6 doesn't support Unpack"
            ),
        ),
        pytest.param(
            Tuple[str, Unpack[Ts]],
            Tuple[Any, ...],
            marks=[
                pytest.mark.skipif(
                    sys.version_info < (3, 7),
                    reason="Python 3.6 doesn't support Unpack",
                ),
                pytest.mark.xfail(
                    HYDRA_VERSION < Version(1, 2, 0),
                    reason="Hydra 1.1.2 doesn't parse tuples beyond first element.",
                ),
            ],
        ),
        pytest.param(
            Tuple[str, Unpack[Ts], int],
            Tuple[Any, ...],
            marks=[
                pytest.mark.skipif(
                    sys.version_info < (3, 7),
                    reason="Python 3.6 doesn't support Unpack",
                ),
                pytest.mark.xfail(
                    HYDRA_VERSION < Version(1, 2, 0),
                    reason="Hydra 1.1.2 doesn't parse tuples beyond first element.",
                ),
            ],
        ),
        (Annotated[int, int], int),
        (Annotated[Tuple[str, str], int], Tuple[str, str]),
        (Annotated[Builds, int], Any),
        (NewType("I", int), int),
        (NewType("S", Tuple[str, str]), Tuple[str, str]),
        (Self, Any),  # type: ignore
        (Literal[1, 2], Any),  # unsupported generics
        (Type[int], Any),
        (Builds, Any),
        (Builds[int], Any),
        (Type[Builds[int]], Any),
        (Set, Any),
        (Set[int], Any),
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
        (Optional[List[Color]], Optional[List[Color]]),
        (
            Optional[List[List[int]]],
            Optional[List[Any]]
            if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES
            else Optional[List[List[int]]],
        ),
        (List[int], List[int]),  # supported containers
        (List[frozenset], List[Any]),
        (
            List[List[int]],
            List[Any] if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES else List[List[int]],
        ),
        (List[Tuple[int, int]], List[Any]),
        (List[T], List[Any]),
        (Dict[str, float], Dict[str, float]),
        (Dict[C, int], Dict[Any, int]),
        (Dict[str, C], Dict[str, Any]),
        (Dict[C, C], Dict[Any, Any]),
        (
            Dict[str, List[int]],
            Dict[str, Any]
            if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES
            else Dict[str, List[int]],
        ),
        (Tuple[str], Tuple[str]),
        (Tuple[str, ...], Tuple[str, ...]),
        (Tuple[str, str, str], Tuple[str, str, str]),
        (
            Tuple[List[int]],
            (
                Tuple[Any]
                if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES
                else Tuple[List[int]]
            ),
        ),
        (Union[NoneType, Tuple[int, int]], Optional[Tuple[int, int]]),
        (Union[Tuple[int, int], NoneType], Optional[Tuple[int, int]]),
        (
            List[Dict[str, List[int]]],
            List[Any]
            if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES
            else List[Dict[str, List[int]]],
        ),
        (
            List[List[Type[int]]],
            List[Any] if not HYDRA_SUPPORTS_NESTED_CONTAINER_TYPES else List[List[Any]],
        ),
        (Tuple[Tuple[int, ...], ...], Tuple[Any, ...]),
        (Optional[Tuple[Tuple[int, ...], ...]], Optional[Tuple[Any, ...]]),
    ],
)
def test_sanitized_type_expected_behavior(in_type, expected_type):

    assert sanitized_type(in_type) == expected_type, in_type

    if in_type != expected_type:
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
    assert sanitized_type(Tuple[int, str, int]) is Tuple[Any, Any, Any]


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
