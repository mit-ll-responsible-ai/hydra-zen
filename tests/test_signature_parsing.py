# Copyright (c) 2021 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import inspect
from abc import ABC
from dataclasses import dataclass
from inspect import Parameter
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings

from hydra_zen import builds, hydrated_dataclass, mutable_value
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


@settings(max_examples=1000)
@given(
    user_value_x=valid_hydra_literals,
    user_value_y=valid_hydra_literals,
    user_value_z=valid_hydra_literals,
    specified_as_default=st.lists(st.sampled_from(["x", "y", "z"]), unique=True),
)
def test_builds_signature_shuffling_takes_least_path(
    user_value_x, user_value_y, user_value_z, specified_as_default
):

    # We will specify an arbitrary selection of x, y, z via `builds`, and then specify the
    # remaining parameters via initializing the resulting dataclass. This ensures that we can
    # accommodate arbitrary "signature shuffling", i.e. that parameters with defaults specified
    # are shuffled just to the right of those without defaults.
    #
    # E.g.
    #  - `builds(f, populate_full_signature=True)`.__init__ -> (x, y, z, has_default=default_value)
    #  - `builds(f, x=1, populate_full_signature=True)`.__init__ -> (y, z, x=1, has_default=default_value)
    #  - `builds(f, y=2, z=-1, populate_full_signature=True)`.__init__ -> (z, y=2, z=-1, has_default=default_value)

    defaults = dict(x=user_value_x, y=user_value_y, z=user_value_z)

    default_override = {k: defaults[k] for k in specified_as_default}
    specified_via_init = {
        k: defaults[k] for k in set(defaults) - set(specified_as_default)
    }

    BuildsF = builds(f2, **default_override, populate_full_signature=True)
    sig_param_names = [p.name for p in inspect.signature(BuildsF).parameters.values()]
    expected_param_ordering = (
        sorted(specified_via_init) + sorted(specified_as_default) + ["has_default"]
    )

    assert sig_param_names == expected_param_ordering

    b = BuildsF(**specified_via_init)
    assert b.x == user_value_x
    assert b.y == user_value_y
    assert b.z == user_value_z
    assert b.has_default == 101


def f3(x: str, *args, y: int = 22, z=[2], **kwargs):
    pass


@pytest.mark.parametrize("include_extra_param", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_builds_with_full_sig_mirrors_target_sig(
    include_extra_param: bool, partial: bool
):

    kwargs = dict(named_param=2) if include_extra_param else {}
    kwargs["y"] = 0  # overwrite default value
    Conf = builds(f3, populate_full_signature=True, hydra_partial=partial, **kwargs)

    params = inspect.signature(Conf).parameters.values()

    expected_sig = [("x", str)] if not partial else []

    expected_sig += [("y", int), ("z", Any)]
    if include_extra_param:
        expected_sig.append(("named_param", Any))

    actual_sig = [(p.name, p.annotation) for p in params]
    assert expected_sig == actual_sig

    if not partial:
        conf = Conf(x=-100)
        assert conf.x == -100
    else:
        # x should be excluded when partial=True and full-sig is populated
        conf = Conf()

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


@pytest.mark.parametrize("target", [a_func, AClass, AMetaClass])
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
        hydra_partial=True
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
