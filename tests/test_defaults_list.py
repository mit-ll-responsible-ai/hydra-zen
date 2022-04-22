# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT

import random
from typing import Any

import hypothesis.strategies as st
import pytest
from hydra.core.config_store import ConfigStore
from hypothesis import given

from hydra_zen import builds, instantiate, launch, make_config
from hydra_zen.errors import HydraZenValidationError


def test_hydra_defaults_work_builds():
    config_store = ConfigStore.instance()
    config_store.store(group="x", name="a", node=builds(int, 10))
    Conf = builds(dict, x=None, y="hi", hydra_defaults=["_self_", {"x": "a"}])
    job = launch(Conf, instantiate)
    assert job.return_value == {"x": 10, "y": "hi"}


def test_hydra_defaults_work_make_config():
    config_store = ConfigStore.instance()
    config_store.store(group="x", name="a", node=builds(int, 10))
    Conf = make_config(x=None, y="hi", hydra_defaults=["_self_", {"x": "a"}])
    job = launch(Conf, instantiate)
    assert job.return_value == {"x": 10, "y": "hi"}


invalid_defaults = st.sampled_from(
    [1, [1], [{"a": 1}], {"a": 1}, {1: "a"}, {1: 1}, False, True]
)
valid_defaults = st.sampled_from(["a", {"a": "b"}, {"a": ["a", "b"]}])


@pytest.mark.filterwarnings("ignore::UserWarning")
@given(
    invalids=invalid_defaults | st.lists(invalid_defaults, min_size=1),
    valids=st.lists(valid_defaults, min_size=0),
    include_self=st.sampled_from([None, "pre", "post"]),
)
def test_hydra_defaults_validation(invalids: Any, valids: list, include_self):
    if isinstance(invalids, list):
        defaults = invalids + valids
        random.shuffle(defaults)
        if include_self == "pre":
            defaults = ["_self_"] + defaults
        elif include_self == "pose":
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
