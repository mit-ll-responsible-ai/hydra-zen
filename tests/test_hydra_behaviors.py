# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, List, Tuple

import hypothesis.strategies as st
import pytest
from hypothesis import given
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import ValidationError

from hydra_zen import (
    builds,
    hydrated_dataclass,
    instantiate,
    make_config,
    mutable_value,
)
from hydra_zen._compatibility import HYDRA_SUPPORTS_OBJECT_CONVERT
from hydra_zen.structured_configs._utils import get_obj_path


def f_three_vars(x, y, z):
    return x, y, z


def wrapper(obj):
    return obj


@given(
    convert=st.sampled_from(["none", "all", "partial"]),
    recursive=st.booleans(),
    sig=st.booleans(),
    partial=st.none() | st.booleans(),
    meta=st.none() | st.just(dict(meta=1)),
    wrappers=st.none() | st.just(wrapper) | st.lists(st.just(wrapper)),
    kwargs=st.dictionaries(keys=st.sampled_from(["x", "y", "z"]), values=st.floats()),
    name=st.none() | st.sampled_from(["NameA", "NameB"]),
)
def test_builds_sets_hydra_params(
    convert, recursive, sig, partial, name, meta, wrappers, kwargs
):
    out = builds(
        f_three_vars,
        hydra_convert=convert,
        hydra_recursive=recursive,
        populate_full_signature=sig,
        zen_partial=partial,
        zen_meta=meta,
        zen_wrappers=wrappers,
        zen_dataclass={"cls_name": name},
        **kwargs,
    )

    assert out._convert_ == convert  # type: ignore
    assert out._recursive_ == recursive  # type: ignore
    if name is not None:
        assert out.__name__ == name
    else:
        assert "Builds_f_three_vars" in out.__name__


def f_for_convert(**kwargs):
    return kwargs


class NotSet:
    pass


@dataclass
class A:
    x: Any = (1, 2)


@pytest.mark.parametrize("via_hydrated_dataclass", [False, True])
@pytest.mark.parametrize(
    "convert, expected_types",
    [
        # "Passed objects are DictConfig and ListConfig"
        ("none", (DictConfig, ListConfig, DictConfig, int)),
        # default should be same as 'none'
        (NotSet, (DictConfig, ListConfig, DictConfig, int)),
        # "Passed objects are converted to dict and list, with
        #  the exception of Structured Configs (and their fields)."
        ("partial", (dict, list, DictConfig, int)),
        # "Passed objects are dicts, lists and primitives without
        # a trace of OmegaConf containers"
        ("all", (dict, list, dict, int)),
    ],
)
def test_hydra_convert(
    convert: str, expected_types: List[type], via_hydrated_dataclass: bool
):
    """Tests that the `hydra_convert` parameter produces the expected/documented
    behavior in hydra."""

    kwargs = dict(hydra_convert=convert) if convert is not NotSet else {}

    if not via_hydrated_dataclass:

        out = instantiate(
            builds(
                f_for_convert,
                a_dict=dict(x=1),
                a_list=[1, 2],
                a_struct_config=A,
                an_int=1,
                zen_convert={"dataclass": False},
                **kwargs,
            )
        )
    else:

        @hydrated_dataclass(f_for_convert, **kwargs)
        class MyConf:
            a_dict: Any = mutable_value(dict(x=1))
            a_list: Any = mutable_value([1, 2])
            a_struct_config: Any = A
            an_int: int = 1

        out = instantiate(MyConf)

    actual_types = tuple(
        type(out[i]) for i in ("a_dict", "a_list", "a_struct_config", "an_int")
    )
    assert actual_types == expected_types

    if convert in ("none", "partial", NotSet):
        expected_field_type = ListConfig
    else:
        expected_field_type = list

    assert type(out["a_struct_config"]["x"]) is expected_field_type


@pytest.mark.xfail(
    not HYDRA_SUPPORTS_OBJECT_CONVERT,
    raises=ValueError,
    reason="Hydra 1.3.0 required for _convert_ = 'object'",
)
def test_object_convert():
    Conf = make_config(
        a=builds(dict, b=1), hydra_convert="object"
    )  # should raise if local Hydra version is < 1.3.0
    assert isinstance(instantiate(Conf), Conf)


@pytest.mark.xfail(
    not HYDRA_SUPPORTS_OBJECT_CONVERT,
    raises=ValueError,
    reason="Hydra 1.3.0 required for _convert_ = 'object'",
)
def test_object_convert_hydrated():
    @hydrated_dataclass(dict, hydra_convert="object")
    class A:
        ...

    assert A._convert_ == "object"  # type: ignore


def f_for_recursive(**kwargs):
    return kwargs


@pytest.mark.parametrize("via_hydrated_dataclass", [False, True])
@pytest.mark.parametrize("recursive", [False, True, NotSet])
def test_recursive(recursive: bool, via_hydrated_dataclass: bool):
    target_path = get_obj_path(f_for_recursive)

    kwargs = dict(hydra_recursive=recursive) if recursive is not NotSet else {}

    if not via_hydrated_dataclass:
        out = instantiate(
            builds(
                f_for_recursive,
                x=builds(f_for_recursive, y=1),
                **kwargs,
            )
        )

    else:

        @hydrated_dataclass(f_for_recursive)
        class B:
            y: int = 1

        @hydrated_dataclass(f_for_recursive, **kwargs)
        class A:
            x: Any = B

        out = instantiate(A)

    if recursive is NotSet or recursive:
        assert out == {"x": {"y": 1}}
    else:
        assert out == {"x": {"_target_": target_path, "y": 1}}


def f(x: Tuple[int]):
    return x


class C:
    def __init__(self, x: int):
        self.x = x


def g(x: C):
    return x


def g2(x: List[C]):
    return x


def test_type_checking():
    conf = builds(f, populate_full_signature=True)(x=("hi",))  # type: ignore
    with pytest.raises(ValidationError):
        instantiate(conf)

    # should be ok
    instantiate(builds(f, populate_full_signature=True)(x=(1,)))

    # nested configs should get validated too
    with pytest.raises(ValidationError):
        instantiate(builds(g, x=builds(C, x="hi")))  # Invalid: `C.x` must be int

    # should be ok
    instantiate(builds(g, x=builds(C, x=1)))

    conf_C = builds(C, x=1)

    # should be ok
    instantiate(builds(g2, x=[conf_C, conf_C]))
