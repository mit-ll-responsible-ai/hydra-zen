# Copyright (c) 2023 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from functools import partial

import pytest

from hydra_zen import (
    DefaultBuilds,
    builds,
    get_target,
    instantiate,
    make_custom_builds_fn,
)
from hydra_zen.funcs import zen_processing
from hydra_zen.typing import HydraPartialBuilds, ZenPartialBuilds

get_obj_path = DefaultBuilds._get_obj_path


@dataclass
class HydraPartialConf:
    _target_: str = get_obj_path(dict)
    _partial_: bool = True
    x: int = 1


@dataclass
class ZenPartialConf:
    _target_: str = get_obj_path(zen_processing)
    _zen_target: str = get_obj_path(dict)
    _zen_partial: bool = True
    x: int = 1


def test_HydraPartialBuilds_protocol():
    assert isinstance(HydraPartialConf(), HydraPartialBuilds)
    assert not isinstance(HydraPartialConf(), ZenPartialBuilds)


def test_HYDRA_SUPPORTS_PARTIAL_is_set_properly():
    obj = instantiate(HydraPartialConf)
    # instantiation should produce `functools.partial(dict, x=1)`
    assert callable(obj) and obj() == {"x": 1}


def test_partial_via_zen_processing():
    # ensures that confs that leverage zen_processing to partial-y
    # instantiate object always works
    obj = instantiate(ZenPartialConf)
    assert isinstance(obj, partial)
    assert obj() == {"x": 1}


def test_get_target_on_hydra_partial_config():
    assert get_target(HydraPartialConf) is dict


@pytest.mark.parametrize("via_custom_builds", [False, True])
def test_builds_leverages_hydra_support_for_partial_when_no_other_zen_processing_used(
    via_custom_builds,
):
    builder = (
        partial(builds, zen_partial=True)
        if not via_custom_builds
        else make_custom_builds_fn(zen_partial=True)
    )
    Conf = builder(dict, x=1)
    instance = instantiate(Conf)()
    assert instance == {"x": 1}


def f(*args, **kwargs):
    return args, kwargs


def identity_wrapper(f):
    return f


@pytest.mark.parametrize("zen_meta", [None, {"_y": 1}])
@pytest.mark.parametrize("zen_wrapper", [(), identity_wrapper])
@pytest.mark.parametrize("pop_sig", [False, True])
def test_partial_func_attr_is_always_target(zen_meta, zen_wrapper, pop_sig):
    # Regardless of whether Hydra performs partial instantiation or if zen_processing
    # is responsible, the result of the instantiation should be a partial object
    # whose `.func` attr is the actual target-object.
    #
    # In particular, we want to ensure that `.func` never points to `zen_processing`,
    # which could still produce the same end behavior for resolving the partial, but
    # would nonetheless have unexpected behaviors to those users who query `.func`.
    Conf = builds(
        f,
        "hi",
        x=1,
        zen_meta=zen_meta,
        zen_wrappers=zen_wrapper,
        populate_full_signature=pop_sig,
        zen_partial=True,
    )

    partial_out = instantiate(Conf)
    assert hasattr(partial_out, "func") and partial_out.func is f
    assert partial_out() == (("hi",), {"x": 1})
