from dataclasses import dataclass
from typing import Any, List

import hypothesis.strategies as st
import pytest
from hypothesis import assume, given
from omegaconf import DictConfig, ListConfig

from hydra_utils import builds, hydrated_dataclass, instantiate, mutable_value
from hydra_utils.structured_configs._utils import get_obj_path


def f_three_vars(x, y, z):
    return x, y, z


@given(
    convert=st.sampled_from(["none", "all", "partial"]),
    recursive=st.booleans(),
    sig=st.booleans(),
    partial=st.booleans(),
    kwargs=st.dictionaries(keys=st.sampled_from(["x", "y", "z"]), values=st.floats()),
    name=st.none() | st.sampled_from(["NameA", "NameB"]),
)
def test_builds_sets_hydra_params(convert, recursive, sig, partial, name, kwargs):
    if partial and not recursive:
        assume(False)

    out = builds(
        f_three_vars,
        hydra_convert=convert,
        hydra_recursive=recursive,
        populate_full_signature=sig,
        hydra_partial=partial,
        dataclass_name=name,
        **kwargs,
    )

    assert out._convert_ == convert
    assert out._recursive_ == recursive
    if name is not None:
        assert out.__name__ == name
    else:
        assert "Builds_f_three_vars" in out.__name__


def f_for_convert(**kwargs):
    return kwargs


class NotSet:
    pass


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

    @dataclass
    class A:
        x: Any = (1, 2)

    kwargs = dict(hydra_convert=convert) if convert is not NotSet else {}

    if not via_hydrated_dataclass:

        out = instantiate(
            builds(
                f_for_convert,
                a_dict=dict(x=1),
                a_list=[1, 2],
                a_struct_config=A,
                an_int=1,
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


def f_for_recursive(**kwargs):
    return kwargs


@pytest.mark.parametrize("via_hydrated_dataclass", [False, True])
@pytest.mark.parametrize("recursive", [False, True])
def test_recursive(recursive: bool, via_hydrated_dataclass: bool):
    target_path = get_obj_path(f_for_recursive)

    if not via_hydrated_dataclass:
        out = instantiate(
            builds(
                f_for_recursive,
                x=builds(f_for_recursive, y=1),
                hydra_recursive=recursive,
            )
        )

    else:

        @hydrated_dataclass(f_for_recursive)
        class B:
            y: int = 1

        @hydrated_dataclass(f_for_recursive, hydra_recursive=recursive)
        class A:
            x: Any = B

        out = instantiate(A)

    if recursive:
        assert out == {"x": {"y": 1}}
    else:
        assert out == {
            "x": {
                "_target_": target_path,
                "_recursive_": True,
                "_convert_": "none",
                "y": 1,
            }
        }
