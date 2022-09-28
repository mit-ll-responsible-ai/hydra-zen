# Copyright (c) 2022 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
import os
import random
import sys
from dataclasses import dataclass

import pytest
from hypothesis import given, strategies as st
from omegaconf import DictConfig

from hydra_zen import builds, make_config, to_yaml, zen
from hydra_zen._zen import Zen
from hydra_zen.errors import HydraZenValidationError
from tests.custom_strategies import everything_except


@zen
def zen_identity(x: int):
    return x


def function(x: int, y: int, z: int = 2):
    return x * y * z


def function_with_args(x: int, y: int, z: int = 2, *args):
    return x * y * z


def function_with_kwargs(x: int, y: int, z: int = 2, **kwargs):
    return x * y * z


def function_with_args_kwargs(x: int, y: int, z: int = 2, *args, **kwargs):
    return x * y * z


def test_zen_basic_usecase():
    @zen
    def f(x: int, y: str):
        return x * y

    Cfg = make_config(x=builds(int, 2), y="cow", unused="unused")
    assert f(Cfg) == 2 * "cow"


@pytest.mark.parametrize("precall", [None, lambda x: x])
def test_zen_wrapper_trick(precall):
    def f(x):
        return x

    # E.g. @zen
    #      def f(...)
    z1 = zen(f, pre_call=precall)  # direct wrap
    assert isinstance(z1, Zen) and z1.func is f and z1.pre_call is precall

    # E.g. @zen(pre_call=...)
    #      def f(...)
    z2 = zen(pre_call=precall)(f)  # config then wrap
    assert isinstance(z2, Zen) and z2.func is f and z1.pre_call is precall


@given(seed=st.sampled_from(range(4)))
def test_zen_precall_precedes_instantiation(seed: int):
    @zen(pre_call=zen(lambda seed: random.seed(seed)))
    def f(x: int, y: int):
        return x, y

    actual = f(
        make_config(
            x=builds(int, 2),
            y=builds(random.randint, 0, 10),
            seed=seed,
        )
    )

    random.seed(seed)
    expected = f.func(2, random.randint(0, 10))

    assert actual == expected


@given(y=st.sampled_from(range(10)), as_dict_config=st.booleans())
def test_interpolations_are_resolved(y: int, as_dict_config: bool):
    @zen
    def f(dict_, list_, builds_, make_config_, direct):
        return dict_, list_, builds_, make_config_, direct

    cfg_maker = make_config if not as_dict_config else lambda **kw: DictConfig(kw)
    out = f(
        cfg_maker(
            dict_={"x": "${y}"},
            list_=["${y}"],
            builds_=builds(dict, a="${y}"),
            make_config_=make_config(b="${y}"),
            direct="${y}",
            y=y,
        )
    )
    assert out == ({"x": y}, [y], {"a": y}, {"b": y}, y)


@pytest.mark.parametrize(
    "cfg",
    [
        dict(x=21),
        DictConfig(dict(x=21)),
        make_config(x=21),
        to_yaml(dict(x=21)),
    ],
)
def test_supported_config_types(cfg):
    @zen
    def f(x):
        return x

    assert f(cfg) == 21


def test_zen_resolves_default_factories():
    Cfg = make_config(x=[1, 2, 3])
    assert zen_identity(Cfg) == [1, 2, 3]
    assert zen_identity(Cfg()) == [1, 2, 3]


def test_zen_works_on_partiald_funcs():
    from functools import partial

    def f(x: int, y: str):
        return x, y

    zen_pf = zen(partial(f, x=1))

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` is missing the following fields: y",
    ):
        zen_pf.validate(make_config())

    zen_pf.validate(make_config(y="a"))
    assert zen_pf(make_config(y="a")) == (1, "a")
    zen_pf.validate(make_config(x=2, y="a"))
    assert zen_pf(make_config(x=2, y="a")) == (2, "a")


class A:
    def f(self, x: int, y: int, z: int = 2):
        return x * y * z


method = A().f


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@pytest.mark.parametrize(
    "cfg",
    [
        make_config(),  # missing x & y
        make_config(not_a_field=2),
        make_config(x=1),  # missing y
        make_config(y=2, z=4),  # missing x
    ],
)
def test_zen_validation_cfg_missing_parameter(cfg, func):
    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` is missing the following fields",
    ):
        zen(func).validate(cfg)


@given(bad_config=everything_except((list, dict, str)))
def test_zen_validate_bad_config(bad_config):
    @zen
    def f(*a, **k):
        ...

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg` Must be a dataclass, dict/DictConfig, list/ListConfig, or yaml-string",
    ):
        f(bad_config)


def test_zen_validation_excluded_param():
    zen(lambda x: ...).validate(make_config(), excluded_params=("x",))


def test_zen_validation_cfg_has_bad_pos_args():
    def f(x):
        return x

    @dataclass
    class BadCfg:
        _args_: int = 1  # bad _args_

    with pytest.raises(
        HydraZenValidationError,
        match=r"`cfg._args_` must be a sequence type",
    ):
        zen(f).validate(BadCfg)


@pytest.mark.parametrize("not_inspectable", [range(1), False])
def test_zen_validate_no_sig(not_inspectable):
    with pytest.raises(
        HydraZenValidationError,
        match="hydra_zen.zen can only wrap callables that possess inspectable signatures",
    ):
        zen(not_inspectable)


@pytest.mark.parametrize(
    "func",
    [
        function,
        function_with_args,
        function_with_kwargs,
        function_with_args_kwargs,
        method,
    ],
)
@given(
    x=st.integers(-10, 10),
    y=st.integers(-10, 10),
    # kwargs=st.dictionaries(st.sampled_from(["z", "not_a_field"]), st.integers()),
    instantiate_cfg=st.booleans(),
)
def test_zen_call(x: int, y: int, instantiate_cfg, func):

    cfg = make_config(x=x, y=y)
    if instantiate_cfg:
        cfg = cfg()

    # kwargs.pop("not_a_field", None)
    expected = func(x, y)
    actual = zen(func)(cfg)
    assert expected == actual


@given(x=st.sampled_from(range(10)))
def test_zen_function_respects_with_defaults(x):
    @zen
    def f(x: int = 2):
        return x

    assert f(make_config()) == 2  # defer to x's default
    assert f(make_config(x=x)) == x  # overwrite x


def raises():
    raise AssertionError("shouldn't have been called!")


@pytest.mark.parametrize(
    "call",
    [
        lambda x: zen_identity(make_config(x=builds(int, x))),
        lambda x: zen_identity(make_config(x=builds(int, x), y=builds(raises))),
    ],
)
@given(x=st.sampled_from(range(10)))
def test_instantiation_only_occurs_as_needed(call, x):
    assert call(x) == x


def test_zen_works_with_non_builds():
    bigger_cfg = make_config(super_conf=make_config(a=builds(int)))
    out = zen(lambda super_conf: super_conf)(bigger_cfg)
    assert out.a == 0


class Pre:
    record = []


pre_call_strat = st.just(lambda cfg: Pre.record.append(cfg.x))


@given(
    pre_call=(pre_call_strat | st.lists(pre_call_strat)),
)
def test_multiple_pre_calls(pre_call):
    Pre.record.clear()
    cfg = make_config(x=1, y="a")
    g = zen_identity.func
    assert zen(pre_call=pre_call)(g)(cfg=cfg) == 1
    assert Pre.record == [1] * (len(pre_call) if isinstance(pre_call, list) else 1)


@pytest.mark.skipif(
    sys.platform.startswith("win") and bool(os.environ.get("CI")),
    reason="Things are weird on GitHub Actions and Windows",
)
@pytest.mark.usefixtures("cleandir")
def test_hydra_main():
    import subprocess
    from pathlib import Path

    from hydra_zen import load_from_yaml

    path = (Path(__file__).parent / "dummy_zen_main.py").absolute()
    assert not (Path.cwd() / "outputs").is_dir()
    subprocess.run(["python", path, "x=1", "y=2"]).check_returncode()
    assert (Path.cwd() / "outputs").is_dir()

    *_, latest_job = sorted((Path.cwd() / "outputs").glob("*/*"))

    assert load_from_yaml(latest_job / ".hydra" / "config.yaml") == {
        "x": 1,
        "y": 2,
        "z": "${y}",
    }
