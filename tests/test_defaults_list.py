# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import random
from functools import partial
from typing import Any, Callable, List

import hypothesis.strategies as st
import pytest
from hydra.core.config_store import ConfigStore
from hypothesis import given
from omegaconf import DictConfig, ListConfig

from hydra_zen import MISSING, builds, instantiate, launch, make_config
from hydra_zen.errors import HydraZenValidationError


@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("fn", [partial(builds, dict), make_config])
@pytest.mark.parametrize(
    "default,overrides",
    [
        ({"x": "a"}, []),
        (DictConfig({"x": "a"}), []),
        (make_config(x="a"), []),
        ({"x": ["a"]}, []),
        ({"x": ListConfig(["a"])}, []),
        ({"x": None}, ["x=a"]),
        ({"x": MISSING}, ["x=a"]),
    ],
)
def test_hydra_defaults_work_as_expected(
    fn: Callable, default: Any, overrides: List[str], version_base
):
    config_store = ConfigStore.instance()
    config_store.store(group="x", name="a", node=builds(int, 10))
    Conf = fn(x=None, y="hi", hydra_defaults=["_self_", default])
    job = launch(Conf, instantiate, **version_base, overrides=overrides)
    assert job.return_value == {"x": 10, "y": "hi"}


invalid_defaults = st.sampled_from(
    [1, [1], [{"a": 1}], {"a": 1}, {1: "a"}, {1: 1}, False, True]
)
valid_defaults = st.sampled_from(
    ["a", {"a": "b"}, {"a": None}, {"a": MISSING}, {"a": ["a", "b"]}]
)


@pytest.mark.filterwarnings("ignore::UserWarning")
@given(
    defaults=st.lists(valid_defaults, min_size=0),
    include_self=st.sampled_from([None, "pre", "post"]),
)
def test_hydra_defaults_validation_passes_on_good_values(defaults: Any, include_self):
    if include_self == "pre":
        defaults = ["_self_"] + defaults
    elif include_self == "post":
        defaults.append("_self_")

    builds(dict, hydra_defaults=defaults)
    make_config(hydra_defaults=defaults)


@pytest.mark.usefixtures("cleandir")
@pytest.mark.filterwarnings("ignore::UserWarning")
@given(
    invalids=invalid_defaults | st.lists(invalid_defaults, min_size=1),
    valids=st.lists(valid_defaults, min_size=0),
    include_self=st.sampled_from([None, "pre", "post"]),
)
def test_hydra_defaults_validation_catches_bad_values(
    invalids: Any, valids: list, include_self
):
    if isinstance(invalids, list):
        defaults = invalids + valids
        random.shuffle(defaults)
        if include_self == "pre":
            defaults = ["_self_"] + defaults
        elif include_self == "post":
            defaults.append("_self_")
    else:
        defaults = invalids

    with pytest.raises(HydraZenValidationError):
        builds(dict, hydra_defaults=defaults)

    with pytest.raises(HydraZenValidationError):
        make_config(hydra_defaults=defaults)


@given(
    selfs=st.lists(st.just("_self_"), min_size=2, max_size=5),
    others=st.lists(st.sampled_from([dict(a="a"), dict(b="b")]), max_size=5),
)
def test_hydra_defaults_raises_multiple_self(selfs: list, others: list):
    defaults = selfs + others
    random.shuffle(defaults)

    with pytest.raises(HydraZenValidationError):
        builds(dict, hydra_defaults=defaults)

    with pytest.raises(HydraZenValidationError):
        make_config(hydra_defaults=defaults)

    with pytest.raises(HydraZenValidationError):
        make_config(defaults=defaults)


def test_no_self_in_defaults_warns():
    with pytest.warns(UserWarning):
        builds(dict, hydra_defaults=[{"a": "b"}])

    with pytest.warns(UserWarning):
        make_config(hydra_defaults=[{"a": "b"}])


def test_redundant_defaults_in_make_config_raises():
    with pytest.raises(TypeError):
        make_config(defaults=["_self_"], hydra_defaults=["_self_"])


def test_regression_284():
    # https://github.com/mit-ll-responsible-ai/hydra-zen/issues/284
    make_config(
        hydra_defaults=[
            "_self_",
            {"model": "resnet"},
            {"decorators_model": None},
            {"decorators_model2": MISSING},
            {"test_transform": "test_transformation_6ch"},
        ]
    )
